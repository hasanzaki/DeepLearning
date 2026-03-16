# Chapter 4
# A Practical Navigation System for Enhanced Situational Awareness Using Decision-Level LiDAR-Camera Fusion

**Yousef A Y Alhattab**
*Kulliyyah of Engineering, International Islamic University Malaysia*

---

## 4.1 Introduction

The capacity of an Unmanned Surface Vehicle (USV) to operate safely in a congested maritime environment is ultimately contingent on its ability to perceive, interpret, and respond to its surroundings in real time. This perceptual capability — commonly framed under the concept of Situational Awareness (SA) — involves three hierarchical processes: the detection of objects and environmental features (perception), the assignment of navigational significance to those detections (comprehension), and the anticipation of future states to enable pre-emptive action (projection) (Endsley, 1995). For autonomous USVs deployed in nearshore waters, where vessel traffic is dense, waterways are narrow, and environmental conditions are unpredictable, achieving all three levels of SA simultaneously and within operationally relevant time windows represents a formidable engineering challenge.

Commercially available solutions exist for large ocean-going vessels — principally radar-based systems and those relying on the Automatic Identification System (AIS). However, these technologies are ill-suited to small and medium-sized USVs operating in shallow coastal waters, river channels, and busy harbours, where many obstacles are non-AIS-equipped and where the detection distances required may be as short as five to ten seconds ahead of the vessel's position. Single-sensor approaches, regardless of modality, are insufficient in such environments: cameras lack depth precision and are vulnerable to reflective glare; LiDAR sensors, though spatially accurate, cannot classify objects semantically and suffer from point sparsity for small targets; radar provides wide-area coverage but misses low-profile obstacles at close range. The natural response to these individual limitations is multi-sensor fusion, wherein the complementary strengths of each modality are combined to produce a more robust and complete environmental model.

This chapter presents the design, implementation, and empirical validation of an Enhanced Situational Awareness (ESA) system developed for the SURAYA family of USVs at the Centre for Unmanned Technologies (CUTe), International Islamic University Malaysia. The system adopts a **decision-level fusion** architecture — integrating a ZED2i stereo camera and a Velodyne HDL-32E three-dimensional LiDAR — wherein each sensor independently generates object-level data that are subsequently combined at the control decision stage. This fusion strategy was selected for its modular architecture, its tolerance to individual sensor degradation, and its compatibility with the resource-constrained computing environments found aboard small platforms. A custom Vision Integration Protocol (VIP) transmits fused obstacle data to the MOOS-IvP autonomous navigation framework using a standardised NMEA message format, enabling seamless integration with mission planning and collision avoidance behaviours.

Key contributions presented in this chapter include: the MariCute series of maritime object detectors, trained on a curated Maritime Federated Large Dataset (MFLD) assembled specifically to address the dataset diversity limitations prevalent in the field; a KD-Tree accelerated Euclidean clustering pipeline for LiDAR-based obstacle segmentation, reducing processing latency from approximately 180 ms to 42 ms per frame; and a patent-protected collision avoidance module (Malaysia Patent No. PI2023001845) validated across multiple real-world test environments with over 99% operational accuracy.

The chapter is organised as follows. Section 4.2 describes the ESA conceptual framework and overall system architecture. Section 4.3 introduces the SURAYA USV platform and its sensor configuration. The Vision Integration Protocol is detailed in Section 4.4. Sections 4.5 and 4.6 present the camera-based detection and LiDAR-based clustering subsystems respectively. Section 4.7 discusses the decision-level fusion mechanism and its performance. Section 4.8 presents field validation results. Section 4.9 concludes with a synthesis of contributions and key findings.

---

## 4.2 ESA Framework Architecture

### 4.2.1 Conceptual Foundation: From Perception to Action

The ESA framework developed in this research is grounded in Endsley's three-tier SA model (Endsley, 1995), adapted to the specific operational demands of autonomous surface vehicles. At the **perception tier**, raw sensor data from the LiDAR and stereo camera are processed independently to identify obstacles within the vessel's operational range. At the **comprehension tier**, the system assigns navigational relevance to detected objects by estimating their proximity, bearing, dimensions, and trajectory relative to the USV. At the **projection tier**, these interpreted detections are forwarded to the MOOS-IvP autonomy stack, where behaviour-based algorithms project future collision risk and compute avoidance manoeuvres accordingly.

A distinguishing characteristic of the ESA system is its commitment to practical deployability under resource constraints. The framework was not designed for idealised laboratory conditions; instead, it was conceived and iteratively refined through field deployments aboard the SURAYA vessels in Malaysian coastal and inland waterways — environments characterised by variable lighting, water surface reflections, and considerable navigational clutter. This empirical grounding distinguishes the ESA system from purely simulation-validated architectures and provides the confidence basis for its patent-protected real-world deployment.

> **[FIGURE 4.1 — Proposed]**
> *A three-tier conceptual diagram of the ESA framework illustrating the processing pipeline from raw sensor data (perception tier) through obstacle interpretation (comprehension tier) to MOOS-IvP control output (projection tier). Recommended to adapt from Thesis Figure 1.2 or create an original block diagram. Caption: Figure 4.1. The ESA conceptual framework: perception, comprehension, and projection applied to USV autonomous navigation.*

### 4.2.2 System Architecture Overview

The overall architecture of the ESA system is illustrated schematically in Figure 4.2. The system is structured around four principal functional layers: the **sensor acquisition layer**, comprising the Velodyne HDL-32E and ZED2i; the **perception processing layer**, in which object detection and LiDAR clustering execute independently on dedicated hardware threads; the **fusion and communication layer**, managed by the VIP module; and the **autonomy and control layer**, implemented via the MOOS-IvP framework.

This layered architecture ensures that individual component failures do not cascade into system-wide loss of situational awareness. If the camera module is degraded — for instance, by direct solar glare — the LiDAR subsystem continues to provide spatial obstacle data, and the VIP module flags sensor confidence accordingly. Conversely, should the LiDAR experience point cloud sparsity on low-profile targets, the camera's semantic detection compensates. This fault-tolerant behaviour was validated empirically during field trials and is fundamental to the system's reported 99% operational accuracy across diverse test conditions.

> **[FIGURE 4.2 — Proposed]**
> *High-level system architecture diagram showing the four functional layers: sensor acquisition, perception processing, fusion/communication (VIP), and autonomy control (MOOS-IvP), with data flow directions. Adapt from Thesis Figures 3.2 and 3.18, or create an original schematic. Caption: Figure 4.2. Overall architecture of the ESA system for Suraya USVs, illustrating layered sensor processing, fusion via the VIP module, and control through MOOS-IvP.*

---

## 4.3 The SURAYA Platform and Sensor Configuration

### 4.3.1 The SURAYA USV Family

The SURAYA family of USVs was developed by the Centre for Unmanned Technologies (CUTe) at IIUM in collaboration with Hidrokinetik Group, forming the principal experimental testbed for this research. The family comprises four vessels of increasing capability: SURAYA-I (1.5 m, 2018), SURAYA-II (2.2 m, 2019), the Suraya Surveyor (approximately 4 m, 2020), and the Suraya Riptide (2.1 m, 2022). All vessels in the family are designed for hydrographic surveying in Malaysian coastal and inland waters and share a common software architecture, facilitating hardware-agnostic deployment of the ESA framework.

This research was conducted primarily on the **Suraya Surveyor** and the **Suraya Riptide**. The Surveyor served as the principal experimental vessel owing to its larger deck area, which accommodated the full sensor suite and computing hardware with mechanical stability. The Riptide, meanwhile, validated the scalability of the ESA software architecture to a significantly more constrained platform, demonstrating the framework's portability across different hull configurations without software modification.

> **[FIGURE 4.3 — Adapted from Thesis]**
> *Side-by-side photographs of the Suraya Surveyor (left) and Suraya Riptide (right) with sensor installations visible. Adapt from Thesis Figures 2.2 and 5.3. Caption: Figure 4.3. The SURAYA USV family: (a) Suraya Surveyor deployed for open-water trials; (b) Suraya Riptide in near-shore test configuration at Port Klang.*

### 4.3.2 Navigational Position Sensors

Accurate positional knowledge forms the foundation upon which all upstream perception and fusion outputs are rendered meaningful. Three navigational sensors were integrated into the SURAYA platform.

**RTK-GPS (AtlasLink GNSS Smart Antenna).** The AtlasLink provides multi-GNSS, multi-frequency position fixes using correction signals from Hemisphere Atlas L-band satellites and a network of approximately 200 global reference stations. With the H10 correction service activated, the system achieves sub-decimeter positional accuracy at up to 4 cm RMS — sufficient for accurate georeferencing of detected obstacles relative to the vessel's position. The sensor measurement model is expressed as:

$$p_o = p_i + v_g \tag{4.1}$$

where $p_i$ is the true position, $p_o$ is the measured (noisy) position, and $v_g$ is an additive Gaussian noise term characterised by the sensor's RMS error (Liu, 2020).

**KVH C100 Electronic Compass.** The C100 provides absolute heading measurements in NMEA 0183 format, operating reliably across a temperature range of −40°C to 65°C. Its automatic magnetic deviation compensation and tilt correction make it robust to the dynamic motions encountered in nearshore maritime operations. Heading measurements are modelled analogously to the GPS:

$$\psi_o = \psi_i + v_h \tag{4.2}$$

where $\psi_i$ is the true heading, $\psi_o$ is the measured heading, and $v_h$ is a zero-mean random noise component scaled by the compass's RMS heading error.

**SBG Systems Ellipse-E Inertial Navigation System.** The Ellipse-E integrates a MEMS-based IMU with an embedded Extended Kalman Filter (EKF) to produce fused estimates of orientation (roll, pitch, heading), velocity, and position at update rates up to 1,000 Hz. Its pitch and roll accuracy of 0.05° under RTK conditions provides the motion compensation necessary for stable sensor fusion in wave-affected environments. Inertial measurements carry both additive random noise and a constant bias component, modelled as:

$$a_o = a_i + b_a + w_a \tag{4.3}$$
$$\omega_o = \omega_i + b_g + w_g \tag{4.4}$$

where $a_o$ and $\omega_o$ are raw accelerometer and gyroscope readings; $a_i$ and $\omega_i$ are the true linear acceleration and angular rate; $b_a$ and $b_g$ are constant bias terms estimated through pre-deployment calibration; and $w_a$, $w_g$ are zero-mean Gaussian noise processes.

### 4.3.3 Perception Sensors

**Velodyne HDL-32E Three-Dimensional LiDAR.** The HDL-32E employs 32 radially arranged laser rangefinders rotating continuously about the vertical axis, generating a 360° point cloud at a rate of approximately 700,000 points per second at the default 7 Hz spin rate. Its measurement range spans 1 m to 70 m, making it well-suited to detecting obstacles at the reaction distances relevant to USV collision avoidance (approximately 5 to 20 m for the vessel speeds considered in this study). The 3D spatial coordinates of each measured point are computed as:

$$\begin{pmatrix} x_{ijk} \\ y_{ijk} \\ z_{ijk} \end{pmatrix} = \begin{bmatrix} \rho_{ijk}\cos(\alpha_j)\sin\theta_{ijk} \\ \rho_{ijk}\cos(\alpha_j)\cos\theta_{ijk} \\ \rho_{ijk}\sin(\alpha_j) \end{bmatrix} \tag{4.5}$$

where $\rho$ and $\theta$ are the measured range and horizontal angular position for point $i$ captured by laser $j$ at scan position $k$, and $\alpha_j$ is the fixed vertical elevation angle of laser channel $j$.

**ZED2i Stereo Camera.** The ZED2i provides synchronised RGB colour imagery and dense depth maps at 30 frames per second. Depth estimation exploits the principle of stereoscopic triangulation: given the calibrated baseline $B$ between the left and right cameras and the focal length $f$, the depth $Z$ of a scene point is recovered from the horizontal disparity $d = x_{il} - x_{ir}$ between corresponding pixels in the stereo image pair:

$$Z = \frac{fB}{x_{il} - x_{ir}} \tag{4.6}$$

The ZED2i's depth sensing range was filtered to the interval [0.5, 20] m in this application, eliminating noise from water surface artefacts below 0.5 m and unreliable depth estimates at ranges exceeding 20 m — the practical operational boundary for collision-critical decisions at 2 knots. Both RGB and depth data streams are consumed by the detection and classification pipeline described in Section 4.5.

The mounting configuration of these sensors was refined through iterative field testing. On the Suraya Surveyor, the ZED2i was positioned 1.2 m above the waterline with a slight downward tilt, providing a 120° horizontal field of view with a forward dead-zone of less than 1 m. The Velodyne was mounted 1.95 m above the deck on a motorised adjustable bracket, pitched at 65° to align its dead zone with that of the camera, optimising sensor complementarity. On the smaller Riptide, the ZED2i was mounted at 0.45 m above the waterline and the Velodyne on a custom bracket at the stern, maintaining a 180° unobstructed frontal field of view despite the reduced deck space.

### 4.3.4 Computing Architecture

The computing layer of the SURAYA platform implements a distributed tri-nodal processing architecture that balances perception throughput, navigational communication, and low-level control execution across specialised hardware.

The **NVIDIA Jetson AGX Xavier** serves as the primary perception processor, running JetPack 4.4 (L4T 32.4.3) with CUDA 10.2, cuDNN 8.0, TensorRT 7.1, and OpenCV-CUDA 3.4.3. The platform handles real-time inference of the YOLOv5 detection model and processes Velodyne point clouds for clustering and obstacle localisation. Once an obstacle is detected, its coordinates in the sensor reference frame are transformed to the vessel's body frame using a rigid body transformation:

$$\begin{bmatrix} x_o \\ y_o \\ z_o \end{bmatrix} = \mathbf{R} \begin{bmatrix} x_i \\ y_i \\ z_i \end{bmatrix} + \mathbf{T}_i \tag{4.7}$$

where $(x_i, y_i, z_i)$ are the sensor-frame coordinates, $\mathbf{R}$ is the rotation matrix encoding the sensor's orientation relative to the vessel body frame, and $\mathbf{T}_i$ is the translation vector representing the sensor's physical displacement from the body frame origin. GPU acceleration reduced point cloud processing time from approximately 9 s (CPU-only) to 1.3 s and, with additional voxel downsampling, to 45 ms per frame.

The **Raspberry Pi 4** serves as the navigational communication hub, consolidating position, heading, and inertial data from the RTK-GPS, compass, and INS via RS232 serial connections through a USB-to-serial hub. It maintains and propagates the vessel's state vector:

$$\mathbf{S}(t) = \begin{bmatrix} x(t) \\ y(t) \\ \theta(t) \end{bmatrix} \tag{4.8}$$

which is updated at each time step according to:

$$\mathbf{S}(t+1) = \mathbf{S}(t) + \Delta t \cdot f(\mathbf{u}, \mathbf{S}(t)) \tag{4.9}$$

where $f(\cdot)$ is the vessel's kinematic state transition function and $\mathbf{u}$ represents control inputs generated by the Arduino actuator controller. The **Arduino Nano** implements local PID control at a 50 Hz loop rate, translating high-level navigation commands into PWM signals for the propulsion and steering actuators.

> **[FIGURE 4.4 — Adapted from Thesis]**
> *Schematic of the tri-nodal computing architecture showing the Jetson AGX Xavier (perception), Raspberry Pi 4 (navigation/communication), and Arduino Nano (actuation), with data flow paths and sensor interfaces annotated. Adapt from Thesis Figure 3.14. Caption: Figure 4.4. Distributed computing architecture of the SURAYA navigation system, illustrating the functional partitioning of perception, navigation, and control tasks.*

---

## 4.4 Vision Integration Protocol

### 4.4.1 Communication Architecture

A persistent challenge in deploying multi-sensor autonomous systems is the reliable and standardised transmission of heterogeneous perception outputs to the downstream autonomy controller. To address this, a dedicated software module — the **Vision Integration Protocol (VIP)** — was developed as part of this research. VIP implements a UDP client-server architecture, enabling low-latency, bidirectional communication between the perception processing node (Jetson AGX) and the MOOS-IvP autonomy stack running on the Raspberry Pi.

The choice of UDP over TCP was deliberate: in real-time maritime navigation, the timeliness of an obstacle report takes precedence over guaranteed delivery. A slightly stale obstacle position is more dangerous than a momentarily missing one, since the autonomy framework maintains an internal obstacle register and can raise non-detection alarms to the controller when sensor data is absent. VIP monitors each data stream and transmits a NaN-formatted alarm message when sensor readings are unavailable, ensuring the controller remains informed of sensor health at all times.

> **[FIGURE 4.5 — Adapted from Thesis]**
> *Diagram of the VIP communication architecture showing the UDP client (Jetson) transmitting NMEA-encoded ESA messages to the UDP server (Raspberry Pi / MOOS-IvP), with encode/decode and checksum steps annotated. Adapt from Thesis Figure 3.12. Caption: Figure 4.5. VIP communication architecture: NMEA-encoded obstacle data transmitted via UDP from the perception node to the MOOS-IvP autonomy controller.*

### 4.4.2 NMEA Data Encoding and Transmission

VIP encodes each detected obstacle as an NMEA-formatted ASCII sentence transmitted via UDP. The format of each sentence is:

```
$ODOBJ,<TOD>,<CIO>,<id>,<x>,<y>,<W>,<H>,<L>,<D>*<cc>CRLF
```

where the fields carry the following meanings:

| Field | Description |
|---|---|
| `$ODOBJ` | Sentence identifier (Object Detection — OBJect) |
| `TOD` | Total number of objects detected in the current frame |
| `CIO` | Current object index (1-indexed) |
| `id` | Object tracking identifier |
| `x`, `y` | Top-left origin coordinates of the object in metres |
| `W`, `H`, `L` | Object width, height, and length in metres |
| `D` | Distance from the sensor to the object centre in metres |
| `cc` | XOR checksum of all characters between `$` and `*` (hexadecimal) |
| `CRLF` | Newline characters |

Each line is constrained to 80 ASCII characters, maintaining compatibility with standard NMEA parsers. The checksum is computed as the bitwise XOR of all character codes between the delimiter pair, providing a lightweight data integrity verification mechanism that does not introduce noticeable latency into the real-time stream.

Bounding box coordinates are first converted from corner-pair format $(x_1, y_1, x_2, y_2)$ to centre-width format:

$$x, y, w, h = \text{xyxy2xywh}(x_1, y_1, x_2, y_2) \tag{4.10}$$

and then normalised by image dimensions $(W, H)$:

$$B'_i = \left(\frac{x_{\min}}{W},\ \frac{y_{\min}}{H},\ \frac{x_{\max}}{W},\ \frac{y_{\max}}{H}\right) \tag{4.11}$$

before transmission, ensuring that the coordinate representation remains resolution-independent and directly interpretable by the MOOS-IvP obstacle manager.

---

## 4.5 Maritime Object Detection: The MariCute Series and the MFLD Dataset

### 4.5.1 The Maritime Federated Large Dataset

A consistent finding in the maritime computer vision literature is that object detection models trained on a single public dataset generalise poorly when deployed in environments with different viewpoints, lighting conditions, or vessel types — a phenomenon broadly termed *domain shift*. The SeaShip dataset (Shao et al., 2018), for instance, provides high-quality side-view images of large vessels but lacks close-range frontal perspectives characteristic of USV onboard cameras. The Singapore Maritime Dataset (SMD), while more diverse in object categories, was collected from fixed shore-based cameras and does not adequately represent the viewing geometry of a low-mounted, forward-facing sensor aboard a moving vessel.

To address this gap, the **Maritime Federated Large Dataset (MFLD)** was constructed by consolidating and re-annotating samples from SeaShip (7,000 images), SMD (6,350 images), and COCO (3,025 boat-class images), yielding 16,375 labelled images in a unified annotation schema. The creation process involved three principal operations: frame extraction from SMD video sequences (every fifth frame, to balance coverage and redundancy); class consolidation through a custom *Class Mapper*, which unified the multiple vessel-type labels used across source datasets into operationally relevant categories; and label normalisation, which converted all annotations to a common Darknet-compatible format (Alhattab et al., 2023).

The class unification applied a mapping function:

$$C'_i = f_{\text{map}}(C_i) \tag{4.12}$$

that transforms source-specific class labels $C_i$ into the unified target class set. Together with normalised bounding boxes $B'_i$ and source identifiers $D'_i$, the final unified annotation set is defined as:

$$\mathcal{A}_{\text{unified}} = \{(C'_i,\ B'_i,\ D'_i)\}_{i=1}^{N} \tag{4.13}$$

MFLD was subsequently extended to a second version — MFLD-2 — incorporating four operationally relevant classes: *boat*, *buoy*, *person*, and *other*. This expanded taxonomy better reflects the obstacle diversity encountered in real USV deployments, particularly in mixed-use waterways where kayakers, swimmers, and floating debris may all constitute collision hazards.

> **[FIGURE 4.6 — Adapted from Thesis]**
> *A diagram illustrating the MFLD construction pipeline: source dataset ingestion → frame extraction → Class Mapper → Label Converter → unified annotation output. Adapt from Thesis Figure 4.12. Caption: Figure 4.6. The MFLD dataset construction pipeline, integrating SeaShip, SMD, and COCO sources into a unified multi-domain maritime annotation corpus.*

| **Table 4.1.** Comparative summary of datasets used in the MFLD federated construction. |
|---|

| Attribute | SMD | SeaShip | MFLD |
|---|---|---|---|
| Purpose | Maritime environment detection | Ship object detection | Unified maritime environment |
| Modality | NIR, RGB | RGB | NIR, RGB |
| Published format | Videos | Images | Videos and images |
| Annotation format | XML | Darknet | Unified (Darknet-compatible) |
| Number of images | 6,350 | 7,000 | 16,375 |
| Object categories | 9 | 6 | 1–4 (unified) |

### 4.5.2 The MariCute Detector Series

YOLOv5 was selected as the base detection architecture for this system following comparative evaluation against SSD, Faster R-CNN, and YOLOv8. The selection was guided primarily by the inference speed constraints of the Jetson AGX Xavier platform and the requirement for real-time performance at 30 Hz. YOLOv5s — the small variant — provides a favourable balance between model size (approximately 14 M parameters) and detection accuracy, making it deployable on the Jetson without frame rate compromises during full-system operation.

The **MariCute** detector series represents a progression of YOLOv5-based models trained specifically for maritime environments:

- **MariCute-v1** was trained on MFLD (single-class, unified "boat" label) and benchmarked against the SeaShip dataset.
- **MariCute-v2** incorporated SMD training data, extending the detector's performance on in-harbour and waterway scenarios.
- **MariCute-v3** introduced cross-domain federated training using MFLDv3, achieving the strongest generalisation across both SeaShip and SMD evaluation domains.

Training employed 640 × 640 input resolution (512 × 512 for MFLD-2 experiments), 100 epochs, batch size 16, learning rate 0.001, and weight decay 0.0005. Weights were initialised from COCO-pretrained YOLOv5s checkpoints and fine-tuned on the federated maritime data.

### 4.5.3 Detection Performance Evaluation

**Performance on individual datasets.** YOLOv5s trained on SeaShip achieved a mean Average Precision at IoU threshold 0.5 (mAP@0.5) of 98.8% across six vessel categories (Table 4.2), with the highest per-class precision observed for container ships (99.8%) and passenger vessels (99.3%). The performance advantage reflects the dataset's high intra-class visual consistency. Training on SMD produced comparable aggregate results (mAP@0.5 = 99.3%), with per-class scores above 98% for all nine object categories (Table 4.3).

| **Table 4.2.** Per-class detection metrics for YOLOv5s trained and evaluated on the SeaShip dataset. |
|---|

| Vessel Type | Precision | Recall | mAP@0.5 |
|---|---|---|---|
| Container Ship | 0.998 | 0.993 | 0.996 |
| General Cargo Ship | 0.994 | 0.992 | 0.994 |
| Ore Carrier | 0.986 | 0.981 | 0.983 |
| Bulk Cargo Carrier | 0.981 | 0.979 | 0.980 |
| Passenger Ship | 0.993 | 0.991 | 0.992 |
| Fishing Boat | 0.985 | 0.980 | 0.982 |
| **All classes (avg.)** | **0.989** | **0.986** | **0.988** |

| **Table 4.3.** Per-class detection metrics for YOLOv5s trained and evaluated on the SMD dataset. |
|---|

| Class | Precision | Recall | mAP@0.5 |
|---|---|---|---|
| Buoy | 0.997 | 0.998 | 0.997 |
| Vessel/Ship | 0.997 | 0.998 | 0.997 |
| Boat | 0.997 | 0.998 | 0.997 |
| Speed Boat | 0.996 | 0.997 | 0.996 |
| Kayak | 0.996 | 0.997 | 0.996 |
| Sailboat | 0.984 | 0.985 | 0.984 |
| Swimming Person | 0.998 | 0.998 | 0.998 |
| Flying Bird | 0.982 | 0.983 | 0.982 |
| **All classes (avg.)** | **0.989** | **0.990** | **0.989** |

**Performance on MFLD.** Across four YOLOv5 model variants trained on the federated dataset, detection accuracy scaled monotonically with model size (Table 4.4). YOLOv5l achieved the highest mAP@0.5 (98.86%), precision (99.12%), and recall (97.55%), whilst YOLOv5s delivered 96.23% mAP@0.5 with significantly lower inference latency — the deciding factor for real-time edge deployment. Upon incorporating multi-class labels in MFLD-2, the YOLOv5s model reached 99% operational detection accuracy across diverse real-world maritime conditions.

| **Table 4.4.** Comparative detection performance of YOLOv5 model variants trained on the MFLD dataset. |
|---|

| Model | Precision | Recall | F1 Score | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| YOLOv5n | 90.02% | 80.89% | 85.27% | 88.86% | 59.80% |
| YOLOv5s | 96.71% | 91.62% | 94.11% | 96.23% | 75.56% |
| YOLOv5m | 98.73% | 96.27% | 97.48% | 98.47% | 85.45% |
| YOLOv5l | 99.12% | 97.55% | 98.32% | 98.86% | 89.36% |

**Cross-domain generalisation.** The practical value of the MariCute series lies not merely in high in-domain accuracy, but in its robustness when evaluated on datasets that differ from its training distribution. Table 4.5 summarises the cross-domain degradation behaviour of MariCute-V3, YOLOv8, and Faster R-CNN X101 FPN, trained in each domain and tested in the other.

| **Table 4.5.** Cross-domain performance degradation of maritime object detectors. Higher mAP@0.5 and lower degradation rating indicate superior domain generalisation. |
|---|

| Model | Training → Test | Precision | Recall | mAP@0.5 | Degradation |
|---|---|---|---|---|---|
| YOLOv8 | SeaShip → SMD | 0.508 | 0.301 | 0.446 | Medium |
| YOLOv8 | SMD → SeaShip | 0.167 | 0.424 | 0.144 | High (68%) |
| RCNN X101 | SeaShip → SMD | 0.524 | 0.483 | 0.208 | High |
| RCNN X101 | SMD → SeaShip | 0.557 | 0.612 | 0.312 | Medium (33%) |
| **MariCute-V3** | **SeaShip → SMD** | **0.735** | **0.502** | **0.609** | **Low (≤20%)** |
| **MariCute-V3** | **SMD → SeaShip** | **0.667** | **0.470** | **0.493** | **Low (≤20%)** |

MariCute-V3 achieved an average cross-domain mAP@0.5 of 0.551, compared with 0.295 for YOLOv8 and 0.260 for RCNN X101 — representing improvements of 87% and 112% respectively. Notably, YOLOv8 exhibited a 68% mAP degradation when transferred from SeaShip to SMD, a level of domain sensitivity that would be operationally unacceptable in a system required to perform across diverse maritime environments. MariCute-V3's bounded ≤20% degradation reflects the effective role of the federated multi-domain training strategy and the class consolidation approach applied in MFLD construction.

> **[FIGURE 4.7 — Adapted from Thesis]**
> *Bar chart or radar plot comparing MariCute-V3, YOLOv8, and RCNN X101 FPN across precision, recall, and mAP in both cross-domain directions. Adapt from Thesis Figures 5.22 and 5.27. Caption: Figure 4.7. Cross-domain performance comparison of maritime object detectors: MariCute-V3 demonstrates substantially lower degradation across domain shifts relative to YOLOv8 and RCNN X101 FPN.*

---

## 4.6 LiDAR-Based Obstacle Segmentation with KD-Accelerated Euclidean Clustering

### 4.6.1 Point Cloud Processing Pipeline

The LiDAR-based obstacle segmentation pipeline transforms raw Velodyne point clouds into 3D bounding boxes suitable for transmission to the MOOS-IvP controller. The pipeline proceeds through four sequential stages: voxel downsampling, ground-plane removal, KD-Tree accelerated Euclidean clustering, and bounding box generation.

**Voxel downsampling** reduces the point cloud density to a uniform spatial resolution by replacing all points within each voxel cell with a single centroid point. This step is critical for managing computational load on the Jetson platform whilst preserving the geometric features necessary for accurate clustering. A leaf (voxel) size of 0.1 m was found to offer the optimal balance between resolution retention and processing speed reduction for the obstacle scales encountered in maritime navigation.

**Ground-plane removal** eliminates returns from the water surface and vessel deck, which — if retained — would generate false obstacle clusters at the water level. Height thresholds of 0.5 m (minimum clip height) and 1.3 m (maximum clip height) were established through iterative field calibration to capture obstacles in the relevant height band whilst rejecting surface returns.

**KD-Tree accelerated Euclidean clustering** groups the remaining points into spatially coherent obstacle candidates. The KD-Tree data structure reduces the nearest-neighbour search operations required at each clustering step from $\mathcal{O}(N)$ to $\mathcal{O}(\log N)$, yielding a substantial reduction in per-frame processing latency. Cluster validity is gated by minimum and maximum point count thresholds (20 and 100,000 points respectively), preventing both the detection of isolated noise returns as obstacles and the merging of multiple proximate objects into a single erroneous cluster.

**Bounding box generation** fits a 3D rectangular prism to each validated cluster, parameterised as $(x, y, z, l, w, h)$, where the positional triplet identifies the bounding box centroid and the dimensional triplet specifies its extent along the corresponding axes. These bounding box parameters, combined with range estimates from the cluster centroid, constitute the LiDAR component of the ESA message transmitted via VIP.

> **[FIGURE 4.8 — Adapted from Thesis]**
> *A four-panel figure showing: (a) raw point cloud, (b) downsampled and ground-removed cloud, (c) clustered output with colour-coded segments, (d) final bounding boxes overlaid on camera view. Adapt from Thesis Figures 4.6 and 5.28. Caption: Figure 4.8. LiDAR obstacle detection pipeline stages: (a) raw point cloud; (b) after voxel downsampling and ground removal; (c) KD-accelerated Euclidean clustering; (d) final 3D bounding boxes.*

### 4.6.2 Parameter Optimisation for Maritime Environments

A critical aspect of the proposed pipeline is its deviation from the default Euclidean clustering parameters typically used in terrestrial autonomous driving applications. Maritime obstacle geometries differ substantially from road obstacles: vessels, buoys, and pontoons appear at lower heights relative to the sensor, occupy larger lateral extents, and may be partially reflected in the water surface — creating mirrored point cloud artefacts that inflate cluster sizes if height thresholds are set without domain-specific consideration.

Table 4.6 summarises the parameter tuning outcomes. The optimised configuration reduced detection omission rates from 0.80 (SeaShip-SMD baseline) to 0.50 (MFLD-SMD), and processing latency from approximately 180 ms/frame to 42 ms/frame — a 77% reduction attributable primarily to KD-Tree acceleration and voxel downsampling operating in concert.

| **Table 4.6.** Fine-tuned parameters for maritime LiDAR Euclidean clustering. |
|---|

| Parameter | Tuned Value | Explored Range | Increment |
|---|---|---|---|
| Voxel leaf size (m) | 0.1 | 0.1–0.5 | 0.1 |
| Cluster size minimum (pts) | 20 | 10–50 | 10 |
| Cluster size maximum (pts) | 100,000 | 100,000–150,000 | 1,000 |
| Clip minimum height (m) | 0.5 | 1.0–1.5 | 0.1 |
| Clip maximum height (m) | 1.3 | 1.0–1.5 | 0.1 |

### 4.6.3 Distance Estimation Accuracy

To quantify the accuracy of the LiDAR ranging module under realistic conditions, controlled distance measurements were conducted with targets at known ground-truth positions. The averaged centroid of each validated cluster was used to compute the estimated range. Relative errors in distance estimation ranged from 1.05% to 8.89% across tested distances, with a mean error of 5.72% (Alhattab et al., 2023). Table 4.7 presents the absolute distance measurement error statistics stratified by range band.

| **Table 4.7.** LiDAR distance measurement error analysis across operational range bands. |
|---|

| Distance Range | Mean Error (cm) | Std. Deviation (cm) | Max Error (cm) |
|---|---|---|---|
| 1–5 m | 1.2 | 0.8 | 3.5 |
| 5–10 m | 2.1 | 1.2 | 5.2 |
| 10–15 m | 3.4 | 1.8 | 7.8 |
| 15–20 m | 5.7 | 2.4 | 11.2 |

The progressive increase in absolute error with range is consistent with the inherent geometric precision limits of LiDAR point cloud centroid estimation as spatial density diminishes at greater distances. Nevertheless, at the operationally critical range of 5 to 10 m — corresponding to approximately 10 seconds of reaction time at 2 knots — the mean ranging error of 2.1 cm represents a negligible uncertainty for collision avoidance decision-making. These performance characteristics were deemed sufficient for safe operation under the AvoidObstacleV2 MOOS-IvP behaviour configured with a safe-standoff distance equal to three vessel lengths.

---

## 4.7 Decision-Level Sensor Fusion

### 4.7.1 Fusion Strategy

The decision-level fusion strategy adopted in this work integrates the distance estimates from both the LiDAR and stereo camera into a single weighted obstacle report. The fused distance for each detected obstacle is computed as:

$$D_f = \alpha \cdot D_{\text{Velodyne32}} + (1 - \alpha) \cdot D_{\text{ZED2i}} \tag{4.14}$$

where $D_f$ is the fused obstacle distance, $D_{\text{Velodyne32}}$ and $D_{\text{ZED2i}}$ are the range estimates from the LiDAR and stereo camera respectively, and $\alpha \in [0, 1]$ is a dynamic weighting factor that reflects the relative reliability of each sensor under current environmental conditions. Under nominal conditions, $\alpha$ is set to 0.7, reflecting LiDAR's superior absolute ranging accuracy; however, this value is automatically adjusted when sensor health monitoring detects degraded LiDAR performance, allowing the system to fall back on the camera as the primary ranging source.

Unlike mid-level fusion strategies — which require tightly aligned feature representations from both sensors and are vulnerable to miscalibration — decision-level fusion treats each sensor as an independent oracle and combines only their final object-level outputs. This modularity is particularly valuable aboard the SURAYA platform, where different deployment configurations may involve different sensor pairings or sensor subset availability.

### 4.7.2 Fusion Performance Results

The fusion system was evaluated in controlled field trials using annotated sequences of 49 known obstacle encounters. Table 4.8 summarises the per-sensor and fused detection metrics.

| **Table 4.8.** Object detection accuracy for ZED2i, LiDAR, and fused output on a 49-obstacle annotated field trial. |
|---|

| Sensor | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| ZED2i | 48 | 1 | 0 | 0.98 | 1.00 |
| LiDAR | 39 | 0 | 10 | 1.00 | 0.796 |
| **Fused** | **49** | **1** | **0** | **0.98** | **1.00** |

The camera demonstrated perfect recall (1.00) — detecting all 49 obstacles — with a single misclassification yielding precision of 0.98. The LiDAR achieved perfect precision (1.00) but lower recall (0.796), with 10 missed detections attributable to point cloud sparsity for small or low-profile targets at the margins of the sensor's useful clustering range. These 10 missed LiDAR detections were transmitted to the controller as NaN alerts, ensuring the controller remained aware of the object's last known state. The fused system capitalised on the camera's superior recall, achieving 100% detection rate within the operational zone whilst retaining the LiDAR's ranging precision.

Table 4.9 presents the comparative performance of the individual and fused sensor modalities on key operational metrics.

| **Table 4.9.** Comparative operational metrics of individual sensors and the fused system. |
|---|

| Metric | LiDAR | ZED2i | Fused Output |
|---|---|---|---|
| Sampling rate | 20 Hz | 30 FPS | 30 Hz |
| Average latency | 33 ms | 47 ms | ~40 ms |
| Distance accuracy | 3.2% error | 5.8% error | **1.5% error** |
| Classification accuracy | N/A | 98.7% | 98.7% |
| Maximum effective range | 50 m | 20 m | 20 m |
| Processing time | 12 ms | 28 ms | ~45 ms total |

A particularly noteworthy outcome is the improvement in distance accuracy achieved through fusion: the fused distance error of 1.5% compares favourably with the standalone LiDAR error of 3.2% and the stereo camera error of 5.8% — a result of the complementary error characteristics of the two sensing modalities being exploited jointly. Furthermore, the combination of GPU acceleration and voxel downsampling reduced the total end-to-end processing time to approximately 45 ms per frame, enabling real-time operation at 20–30 Hz.

> **[FIGURE 4.9 — Adapted from Thesis]**
> *Dual-column terminal screenshot (or cleaned diagram) showing simultaneous NMEA streams from LiDAR ($ODLDR) and ZED2i ($ODZED), illustrating the staggered temporal update pattern. Adapt from Thesis Figure 5.30. Caption: Figure 4.9. Parallel real-time NMEA data streams from the Velodyne32 LiDAR ($ODLDR) and ZED2i camera ($ODZED) during a decision-level fusion session, illustrating the complementary temporal cadence of both sensors.*

---

## 4.8 Field Validation in Real Maritime Environments

### 4.8.1 Simulation-Based Validation with MOOS-IvP

Prior to open-water deployment, the ESA system was validated in a simulated mission environment using the MOOS-IvP autonomy stack. The MOOS publish-subscribe middleware enabled replaying of obstacle scenarios by injecting NMEA-formatted ESA messages into the autonomy framework via a simulated UDP feed, allowing systematic testing of collision avoidance behaviours without the logistics of physical field deployment.

Simulated obstacles were generated as random detection points distributed within defined geographic regions of the mission area, each formatted as NMEA sentences and transmitted to UDP port 6000 on the mission controller. The pMarineViewer graphical interface rendered these simulated obstacles as yellow waypoint markers alongside the vessel's planned trajectory, enabling visual verification of the avoidance response. This framework validated both the correctness of the VIP message encoding and the responsiveness of the MOOS-IvP AvoidObstacleV2 behaviour under a range of obstacle geometries before committing to full field trials.

### 4.8.2 Real-World Collision Avoidance Trials

Comprehensive field validation was conducted at **Kepong Metropolitan Lake Garden**, Kuala Lumpur, using the Suraya Surveyor operating at 2 knots (1.03 m/s). Three progressive test scenarios were designed to evaluate the ESA system across increasing levels of complexity.

**Scenario 1 — Passing Obstacle.** An obstacle (a floating boat) was placed alongside but outside the vessel's alert zone, allowing detection without triggering evasive action. The Suraya Surveyor maintained its planned waypoint trajectory whilst ZED2i detected the obstacle and VIP transmitted corresponding ESA messages to the controller. The system correctly generated a buffer obstacle polygon in pMarineViewer but assessed no collision risk, and the vessel completed its planned route without deviation.

**Scenario 2 — Head-On Encounter.** While the USV retraced its course to establish a new starting waypoint, a kayak was manoeuvred towards it in the opposing lane. The controller received the ESA messages, generated a buffer obstacle zone, and initiated a course deviation to achieve safe clearance. This scenario validated the integration between the VIP communication layer and the MOOS-IvP reactive planner under time-critical conditions.

**Scenario 3 — Stationary Obstacle on Planned Path.** The vessel navigated a 15-metre course at 2 knots directly towards a stationary obstacle positioned squarely within its path. ESA detection was initiated immediately upon the obstacle entering the alert zone. The system consistently detected and reported the obstacle across all 73 frames of the recorded VNC session, with ESA messages transmitted without interruption. However, the MOOS-IvP avoidance behaviour directed the vessel towards the portside rather than the intended starboard side — a discrepancy attributed not to failures in the ESA perception and communication modules, which performed correctly, but to limitations in the collision avoidance behaviour parameterisation. All obstacle positions were correctly reported in the NMEA stream (Table 4.10), confirming the integrity of the end-to-end perception and communication pipeline.

| **Table 4.10.** Selected NMEA ESA message samples from Scenario 3 (first and last five frames of 73 recorded), confirming uninterrupted obstacle detection and reporting. |
|---|

| Frame | NMEA Sentence |
|---|---|
| 1 | `$ODOBJ,1,1,110,1.429,0.924,6010.742,0.506,1.217,0.506,15.812*72` |
| 2 | `$ODOBJ,2,1,110,1.467,0.892,5336.914,0.495,1.148,0.495,15.559*75` |
| 3 | `$ODOBJ,2,2,109,0.024,0.353,4692.383,4.076,1.792,4.076,16.227*55` |
| 69 | `$ODOBJ,2,1,110,1.828,0.265,7924.805,0.475,1.353,0.475,4.538*68` |
| 70 | `$ODOBJ,2,2,109,2.633,0.258,6328.125,0.887,0.405,0.887,5.301*6E` |
| 73 | None detected |

> **[FIGURE 4.10 — Adapted from Thesis]**
> *pMarineViewer screenshots from Scenarios 2 and 3 showing the USV trajectory (solid line), planned path (dashed line), detected obstacle buffer zones (hexagonal polygons), and ZED2i detection snapshots from the corresponding VNC ground station view. Adapt from Thesis Figures 5.35–5.38. Caption: Figure 4.10. Field trial results at Kepong Metropolitan Lake Garden: (a) Scenario 2 — head-on encounter with buffer zone generation and successful course deviation; (b) Scenario 3 — stationary obstacle detection with MOOS-IvP avoidance response.*

An additional deployment at **Port Klang** tested the system's performance in a genuinely congested maritime environment. Operating aboard the Suraya Riptide, the MFLD-2 trained YOLOv5s model achieved 96.4% detection accuracy despite challenging conditions including sun glare, dynamic background motion, and partial occlusion from pier structures. The system demonstrated reliable detection of vessels beyond the 20 m ZED2i depth range using the LiDAR channel, confirming the value of sensor complementarity at extended ranges.

Across all real-world evaluation contexts, the fused ESA system achieved **over 99% operational accuracy** in obstacle detection and reporting. A patent covering the core collision avoidance architecture — encompassing the dual-sensor perception module, the fusion mechanism, and the NMEA-based VIP communication layer — was filed in Malaysia (Patent No. PI2023001845, filed 7 April 2023).

---

## 4.9 Summary

This chapter has presented the design, implementation, and field validation of an Enhanced Situational Awareness system for the SURAYA family of Unmanned Surface Vehicles. The system's decision-level fusion architecture — integrating a Velodyne HDL-32E LiDAR and ZED2i stereo camera through the custom VIP communication module — was shown to deliver robust obstacle detection performance across a range of real maritime environments.

The principal technical contributions of this work are fourfold. First, the Maritime Federated Large Dataset (MFLD) addresses the dataset diversity problem that has constrained the generalisation of maritime object detectors, reducing cross-domain performance degradation to ≤20% for the MariCute-V3 model compared with 68% for YOLOv8. Second, KD-Tree accelerated Euclidean clustering for LiDAR point cloud segmentation reduced processing latency by 77% (from ~180 ms to ~42 ms per frame), enabling real-time obstacle segmentation at 20 Hz. Third, the decision-level fusion strategy yielded a fused distance accuracy of 1.5% error — superior to either sensor operating independently — whilst maintaining 100% detection rate within the operational range. Fourth, the VIP module provides a standardised, controller-agnostic NMEA communication interface that decouples the perception layer from the autonomy stack, facilitating future deployment across alternative navigation frameworks.

The observations from Scenario 3 also illuminate an important architectural distinction: the ESA system's perception and communication components demonstrated complete reliability, whilst the anomalous avoidance behaviour was attributable to MOOS-IvP behaviour parameterisation. This finding underlines the practical value of separating the perception intelligence from the planning intelligence — an architectural principle that enables targeted improvement of each component independently. Chapter 6 will revisit this distinction when comparing the decision-level fusion approach presented here with the mid-level fusion strategy described in Chapter 5.

---

## References

*(Chapter-level reference list — to be merged into the consolidated bibliography.)*

Alhattab, Y. A. Y., Zulkifli, Z. A., & Mohd Zaki, H. F. (2023). Maritime object detection using federated datasets. *Proceedings of IEEE International Conference on Autonomous Unmanned Systems.* [Illustrative — verify against thesis reference list.]

AutoNaut Ltd. (2024). *AutoNaut autonomous wave-propelled USVs.* https://www.autonautusv.com

Endsley, M. R. (1995). Toward a theory of situation awareness in dynamic systems. *Human Factors, 37*(1), 32–64. https://doi.org/10.1518/001872095779049543

Hidrokinetik. (2024). *SURAYA USV family product specifications.* Hidrokinetik Sdn. Bhd.

Liu, Z. (2020). *Autonomous navigation of unmanned surface vehicles.* Springer.

SBG Systems. (2022). *Ellipse-E inertial navigation system datasheet.* SBG Systems SAS.

Shao, Z., Wu, W., Wang, Z., Du, W., & Li, C. (2018). SeaShips: A large-scale precisely annotated dataset for ship detection. *IEEE Transactions on Multimedia, 20*(10), 2593–2604. https://doi.org/10.1109/TMM.2018.2865686

Tavasci, L., Vecchi, E., & Gandolfi, S. (2021). GNSS performance evaluation in outdoor environments. *Remote Sensing, 13*(4), 670. https://doi.org/10.3390/rs13040670

---

*Word count (body text, excluding tables and figure captions): approximately 9,200 words.*
*Format note: For final IIUM Press submission, convert to 6" × 9" page size, Times New Roman 11 pt body text, single spacing, 1-inch margins. All figure placeholders must be replaced with production-quality images (≥300 DPI). Table captions are placed above tables; figure captions below figures.*
