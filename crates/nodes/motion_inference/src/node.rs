use std::{collections::VecDeque, future::Future, pin::Pin, sync::Arc};

use anyhow::anyhow;
use booster::{JointsMotorState, LowState};
use color_eyre::Result;
use kinematics::joints::Joints;
use nalgebra::UnitQuaternion;
use ros_z::{
    Message, ServiceTypeInfo,
    entity::TypeInfo,
    message::Service,
    prelude::*,
    pubsub::Received,
    qos::{QosDurability, QosHistory},
    service::ServiceReply,
    time::Time,
};
use serde::{Deserialize, Serialize};
use tokio::{task::JoinHandle, time::Instant};
use types::time_wrapper::TimeWrapper;

use crate::{
    config::Policy,
    inference::{Inference, InferenceCommand, InferenceOutput},
    observation::{self, SensorFrame, VelocityEstimator, validate_time},
};

pub const SENSOR_TOPIC: &str = "inputs/low_state";
pub const TARGETS_TOPIC: &str = "collected_target_joint_positions";
pub const INFERENCE_SERVICE: &str = "motion_inference/infer";
pub const STATUS_TOPIC: &str = "motion_inference/status";

pub struct Infer;

impl Service for Infer {
    type Request = Request;
    type Response = Response;
}

impl ServiceTypeInfo for Infer {
    fn service_type_info() -> TypeInfo {
        let descriptor = ros_z_schema::ServiceDef::new(
            "motion_inference::node::Infer",
            Request::type_name(),
            Response::type_name(),
        )
        .expect("static inference service descriptor is valid");
        let hash = ros_z_schema::compute_hash(&descriptor).expect("static service hash is valid");
        TypeInfo::new(descriptor.type_name.as_str(), hash)
    }
}

pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(run(ctx))
}

async fn run(ctx: Arc<Context>) -> Result<()> {
    let node = ctx.create_node("motion_inference").build().await?;
    let parameters = node.bind_parameter_as::<Parameters>("motion_inference")?;
    let snapshot = Arc::new(parameters.snapshot().typed().clone());
    snapshot
        .validate()
        .map_err(|error| color_eyre::eyre::eyre!("{error:#}"))?;
    let frozen = snapshot.clone();
    parameters.add_validation_hook(move |candidate| {
        if candidate != frozen.as_ref() {
            return Err("motion inference parameters are startup-only; edit startup configuration and restart the process".into());
        }
        Ok(())
    })?;
    let mut runtime = InferenceNode::new(snapshot);
    runtime.start_initialization()?;
    runtime.run(node).await
}

struct QueuedRequest {
    request: Request,
    reply: ServiceReply<Infer>,
    deadline: Instant,
}

#[derive(Clone, Copy)]
enum Job {
    Initialize,
    Inference {
        request_time: Time,
        sensor_time: Time,
        started: Time,
    },
}

enum Completion {
    Initialized,
    Inference(Output),
}

struct Worker {
    handle: JoinHandle<(Controller, anyhow::Result<Completion>)>,
    job: Job,
    deadline: Instant,
    reply: Option<ServiceReply<Infer>>,
    response_sent: bool,
}

struct InferenceNode {
    parameters: Arc<Parameters>,
    controller: Option<Controller>,
    worker: Option<Worker>,
    pending: VecDeque<QueuedRequest>,
    first_fault: Option<String>,
    sensor: Option<SensorFrame>,
    targets: Option<Received<Joints<f32>>>,
    velocity: VelocityEstimator,
}

impl InferenceNode {
    fn new(parameters: Arc<Parameters>) -> Self {
        Self {
            controller: Some(Controller::new(parameters.clone())),
            parameters,
            worker: None,
            pending: VecDeque::new(),
            first_fault: None,
            sensor: None,
            targets: None,
            velocity: VelocityEstimator::default(),
        }
    }

    fn initialized(&self) -> bool {
        self.controller
            .as_ref()
            .is_some_and(Controller::initialized)
            || self
                .worker
                .as_ref()
                .is_some_and(|worker| matches!(worker.job, Job::Inference { .. }))
    }

    fn start_initialization(&mut self) -> Result<()> {
        let deadline = monotonic_deadline(self.parameters.timing.initialization_timeout)?;
        let mut controller = self.controller.take().expect("startup controller exists");
        let handle = tokio::task::spawn_blocking(move || {
            let result = if Instant::now() >= deadline {
                Err(anyhow!("initialization worker expired in the queue"))
            } else {
                controller.initialize().map(|()| Completion::Initialized)
            };
            (controller, result)
        });
        self.worker = Some(Worker {
            handle,
            job: Job::Initialize,
            deadline,
            reply: None,
            response_sent: false,
        });
        Ok(())
    }

    async fn run(&mut self, node: Node) -> Result<()> {
        let latest = QosProfile {
            history: QosHistory::from_depth(1),
            ..Default::default()
        };
        let sensors = node
            .subscriber::<LowState>(SENSOR_TOPIC)
            .qos(latest)
            .build()
            .await?;
        let targets = node
            .subscriber::<Joints<f32>>(TARGETS_TOPIC)
            .qos(latest)
            .build()
            .await?;
        let mut requests = node
            .service_server::<Infer>(INFERENCE_SERVICE)
            .qos(QosProfile {
                history: QosHistory::KeepAll,
                ..Default::default()
            })
            .build()
            .await?;
        let statuses = node
            .publisher::<Status>(STATUS_TOPIC)
            .qos(QosProfile {
                durability: QosDurability::TransientLocal,
                ..Default::default()
            })
            .build()
            .await?;
        statuses
            .publish(&Status {
                time: node.clock().now(),
                state: State::Idle,
            })
            .await?;

        loop {
            if let Some(reason) = self.first_fault.clone() {
                if let Some(worker) = &mut self.worker
                    && !worker.response_sent
                {
                    worker.response_sent = true;
                    if let Some(reply) = worker.reply.take()
                        && let Job::Inference { request_time, .. } = worker.job
                    {
                        respond(
                            reply,
                            request_time,
                            InferenceResult::RequestDenied {
                                reason: reason.clone(),
                            },
                        )
                        .await;
                    }
                }
                self.reject_pending(&reason).await;
            }
            let deadline = self
                .pending
                .iter()
                .map(|request| request.deadline)
                .chain(
                    self.worker
                        .iter()
                        .filter(|worker| !worker.response_sent)
                        .map(|worker| worker.deadline),
                )
                .min();
            tokio::select! {
                biased;
                completed = async { (&mut self.worker.as_mut().expect("worker exists").handle).await }, if self.worker.is_some() => {
                    let mut worker = self.worker.take().expect("worker completed");
                    let (controller, result) = match completed {
                        Ok(completed) => completed,
                        Err(error) => {
                            let reason = format!("motion inference worker failed: {error}");
                            if !worker.response_sent {
                                self.fail_job(&node, &statuses, worker.job, worker.reply.take(), &reason).await?;
                            }
                            self.reject_pending(&reason).await;
                            return Err(color_eyre::eyre::eyre!(reason));
                        }
                    };
                    self.controller = Some(controller);
                    if worker.response_sent { continue; }
                    if Instant::now() >= worker.deadline {
                        self.fail_job(&node, &statuses, worker.job, worker.reply.take(), "motion inference worker timed out").await?;
                        continue;
                    }
                    match result {
                        Ok(Completion::Initialized) => {
                            self.sensor = None;
                            self.targets = None;
                            self.velocity = VelocityEstimator::default();
                            statuses.publish(&Status { time: node.clock().now(), state: State::Initialized }).await?;
                        }
                        Ok(Completion::Inference(output)) => {
                            let Job::Inference { sensor_time, started, .. } = worker.job else { unreachable!("inference completion belongs to inference job") };
                            let now = node.clock().now();
                            if now < started || now >= output.valid_until {
                                self.fail_job(&node, &statuses, worker.job, worker.reply.take(), "inference joints expired or clock moved backwards during execution").await?;
                            } else {
                                respond(worker.reply.take().expect("active request has a reply"), sensor_time, InferenceResult::Output(output)).await;
                            }
                        }
                        Err(error) => self.fail_job(&node, &statuses, worker.job, worker.reply.take(), &format!("{error:#}")).await?,
                    }
                }
                () = async { tokio::time::sleep_until(deadline.expect("deadline exists")).await }, if deadline.is_some() => {
                    let now = Instant::now();
                    if let Some(worker) = &mut self.worker && !worker.response_sent && now >= worker.deadline {
                        worker.response_sent = true;
                        let job = worker.job;
                        let reply = worker.reply.take();
                        self.fail_job(&node, &statuses, job, reply, "motion inference worker timed out").await?;
                    }
                    while let Some(index) = self.pending.iter().position(|request| now >= request.deadline) {
                        let request = self.pending.remove(index).expect("expired request exists");
                        deny(request.reply, &request.request, "request expired in queue").await;
                    }
                }
                received = sensors.recv_with_metadata() => {
                    let received = received?;
                    if self.initialized() && self.first_fault.is_none() {
                        let result = sensor_frame(&received.message, received.source_time, Joints::default()).and_then(|sensor| {
                            sensor.validate(node.clock().now(), &self.parameters)?;
                            self.velocity.update(&sensor, &self.parameters)?;
                            Ok(sensor)
                        });
                        match result {
                            Ok(sensor) => self.sensor = Some(sensor),
                            Err(error) => statuses.publish(&fault_status(&mut self.first_fault, node.clock().now(), error)).await?,
                        }
                    }
                }
                received = targets.recv_with_metadata() => {
                    let received = received?;
                    if self.targets.as_ref().is_none_or(|old| received.source_time >= old.source_time) {
                        self.targets = Some(received);
                    }
                }
                received = requests.take_request_async() => {
                    let (request, reply) = received?.into_parts();
                    if let Some(reason) = &self.first_fault {
                        deny(reply, &request, reason).await;
                    } else if !self.initialized() {
                        deny(reply, &request, "inference is initializing").await;
                    } else {
                        let now = node.clock().now();
                        let age = self.parameters.timing.maximum_input_age;
                        if let Err(error) = validate_time(now, request.time, "inference request", age) {
                            deny(reply, &request, &format!("{error:#}")).await;
                            continue;
                        }
                        let deadline = monotonic_deadline((request.time + age).duration_since(now))?;
                        self.pending.push_back(QueuedRequest { request, reply, deadline });
                    }
                }
                () = std::future::ready(()), if self.worker.is_none() && !self.pending.is_empty() => {
                    let queued = self.pending.pop_front().expect("queued request exists");
                    let request = queued.request;
                    let now = node.clock().now();
                    let maximum_age = self.parameters.timing.maximum_input_age;
                    if Instant::now() >= queued.deadline || now >= request.time + maximum_age {
                        deny(queued.reply, &request, "request expired in queue").await;
                        continue;
                    }
                    let (Some(sensor), Some(targets)) = (&self.sensor, &self.targets) else {
                        deny(queued.reply, &request, "sensor frame or composed targets are missing").await;
                        continue;
                    };
                    let mut sensor = sensor.clone();
                    sensor.last_commanded_position = targets.message;
                    let validation = sensor.validate(now, &self.parameters)
                        .and_then(|()| validate_time(now, targets.source_time, "composed joint targets", maximum_age))
                        .and_then(|()| validate_time(now, request.time, "inference request", maximum_age));
                    if let Err(error) = validation {
                        let reason = format!("{error:#}");
                        statuses.publish(&fault_status(&mut self.first_fault, now, error)).await?;
                        deny(queued.reply, &request, &reason).await;
                        continue;
                    }
                    let valid_until = sensor.timestamp.min(targets.source_time).min(request.time) + maximum_age;
                    if now >= valid_until {
                        let reason = "inference joints expired before execution";
                        statuses.publish(&fault_status(&mut self.first_fault, now, anyhow!(reason))).await?;
                        deny(queued.reply, &request, reason).await;
                        continue;
                    }
                    let deadline = queued.deadline.min(monotonic_deadline(valid_until.duration_since(now))?);
                    let job = Job::Inference { request_time: request.time, sensor_time: sensor.timestamp, started: now };
                    let velocity = self.velocity.clone();
                    let mut controller = self.controller.take().expect("idle controller exists");
                    let clock = node.clock().clone();
                    let handle = tokio::task::spawn_blocking(move || {
                        let started = clock.now();
                        let result = if Instant::now() >= deadline || started >= valid_until || started < now {
                            Err(anyhow!("request expired before worker execution or clock moved backwards"))
                        } else {
                            controller.execute(started, &sensor, request.inner, velocity)
                                .map(|inference| Completion::Inference(Output { inference, valid_until }))
                        };
                        (controller, result)
                    });
                    self.worker = Some(Worker { handle, job, deadline, reply: Some(queued.reply), response_sent: false });
                }
            }
        }
    }

    async fn reject_pending(&mut self, reason: &str) {
        while let Some(queued) = self.pending.pop_front() {
            deny(queued.reply, &queued.request, reason).await;
        }
    }

    async fn fail_job(
        &mut self,
        node: &Node,
        statuses: &Publisher<Status>,
        job: Job,
        reply: Option<ServiceReply<Infer>>,
        reason: &str,
    ) -> Result<()> {
        statuses
            .publish(&fault_status(
                &mut self.first_fault,
                node.clock().now(),
                anyhow!("{reason}"),
            ))
            .await?;
        if let (Job::Inference { request_time, .. }, Some(reply)) = (job, reply) {
            respond(
                reply,
                request_time,
                InferenceResult::RequestDenied {
                    reason: reason.to_owned(),
                },
            )
            .await;
        }
        Ok(())
    }
}

fn monotonic_deadline(duration: std::time::Duration) -> Result<Instant> {
    Instant::now()
        .checked_add(duration)
        .ok_or_else(|| color_eyre::eyre::eyre!("deadline exceeds monotonic clock range"))
}

async fn respond(reply: ServiceReply<Infer>, time: Time, result: InferenceResult) {
    let _ = reply
        .reply_async(&Response {
            time,
            inner: result,
        })
        .await;
}

async fn deny(reply: ServiceReply<Infer>, request: &Request, reason: &str) {
    respond(
        reply,
        request.time,
        InferenceResult::RequestDenied {
            reason: reason.to_owned(),
        },
    )
    .await;
}

fn fault_status(first_fault: &mut Option<String>, time: Time, error: anyhow::Error) -> Status {
    let reason = first_fault
        .get_or_insert_with(|| format!("{error:#}"))
        .clone();
    Status {
        time,
        state: State::Fault { reason },
    }
}

mod messages {
    use super::*;

    pub type Request = TimeWrapper<InferenceCommand>;
    pub type Response = TimeWrapper<InferenceResult>;

    #[derive(Clone, Serialize, Deserialize, Message)]
    pub enum InferenceResult {
        Output(Output),
        RequestDenied { reason: String },
    }

    #[derive(Clone, Serialize, Deserialize, Message)]
    pub struct Status {
        pub time: Time,
        pub state: State,
    }

    #[derive(Clone, Serialize, Deserialize, Message)]
    pub enum State {
        Idle,
        Initialized,
        Fault { reason: String },
    }

    #[derive(Clone, Serialize, Deserialize, Message)]
    pub struct Output {
        pub inference: InferenceOutput,
        pub valid_until: Time,
    }
}

pub use crate::config::Parameters;
pub use messages::{InferenceResult, Output, Request, Response, State, Status};

fn sensor_frame(
    low_state: &LowState,
    timestamp: Time,
    last_targets: Joints<f32>,
) -> anyhow::Result<observation::SensorFrame> {
    let motors = low_state
        .serial_motor_states()
        .map_err(|error| anyhow!("{error:#}"))?;
    let angles = low_state.imu_state.roll_pitch_yaw;
    let orientation = UnitQuaternion::from_euler_angles(angles.x(), angles.y(), angles.z());
    Ok(observation::SensorFrame {
        timestamp,
        position: motors.positions(),
        velocity: motors.velocities(),
        orientation: orientation.into_inner(),
        gyro: low_state.imu_state.angular_velocity,
        last_commanded_position: last_targets,
    })
}

struct Controller {
    parameters: Arc<Parameters>,
    inference: Option<Inference>,
}

impl Controller {
    fn new(parameters: Arc<Parameters>) -> Self {
        Self {
            parameters,
            inference: None,
        }
    }

    fn initialized(&self) -> bool {
        self.inference.is_some()
    }

    fn initialize(&mut self) -> anyhow::Result<()> {
        self.inference = Some(Inference::new(
            &self.parameters.neural_networks_folder,
            &Policy::ALL,
            self.parameters.clone(),
        )?);
        Ok(())
    }

    fn execute(
        &mut self,
        now: Time,
        sensor: &SensorFrame,
        command: InferenceCommand,
        velocity: VelocityEstimator,
    ) -> anyhow::Result<InferenceOutput> {
        let inference = self
            .inference
            .as_mut()
            .ok_or_else(|| anyhow!("inference is initializing"))?;
        inference.execute_request(now, sensor, command, velocity)
    }
}
