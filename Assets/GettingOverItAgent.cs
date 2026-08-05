using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Demonstrations;
using Unity.MLAgents.Sensors;
using UnityEngine;

[RequireComponent(typeof(PlayerControl))]
public class GettingOverItAgent : Agent {
    public const int ObservationSize = 20;
    public const int ContinuousActionSize = 2;
    const int RaycastBufferSize = 16;

    [Header("Environment references")]
    [SerializeField] PlayerControl playerControl;
    [SerializeField] Rigidbody2D body;
    [SerializeField] Rigidbody2D hammer;
    [SerializeField] Camera followCamera;
    [SerializeField] DemonstrationRecorder demonstrationRecorder;

    [Header("Observation normalization")]
    [SerializeField] float positionScale = 10.0f;
    [SerializeField] float bodyVelocityScale = 6.0f;
    [SerializeField] float hammerVelocityScale = 10.0f;
    [SerializeField] float heightScale = 6.5f;
    [SerializeField] float rayLength = 2.0f;
    [SerializeField] LayerMask terrainMask = 1;

    [Header("Episode")]
    [SerializeField] float goalY = 6.5f;
    [SerializeField] float fallY = -4.0f;
    [SerializeField] float stepPenalty = -0.0001f;
    [SerializeField] float successReward = 10.0f;
    [SerializeField] float fallPenalty = -1.0f;

    [Header("Waypoint curriculum")]
    [SerializeField] RLWaypoint[] waypoints = new RLWaypoint[0];
    [SerializeField] float heightProgressScale = 0.25f;
    [SerializeField] float waypointProgressScale = 0.5f;
    [SerializeField] float waypointReward = 1.0f;
    [SerializeField] float finalWaypointReward = 10.0f;
    [SerializeField] bool terminateAtFinalWaypoint = true;

    Vector2 initialBodyPosition;
    float initialBodyRotation;
    Vector2 initialHammerPosition;
    float initialHammerRotation;
    Vector3 initialCameraPosition;
    Vector2 previousAction;
    float maximumHeight;
    int activeWaypointIndex;
    float previousWaypointDistance;
    bool initialStateCached;
    bool waypointConfigurationValidated;
    Collider2D hammerCollider;
    readonly RaycastHit2D[] raycastResults =
        new RaycastHit2D[RaycastBufferSize];

    bool CommunicatorActive {
        get {
            return Academy.IsInitialized &&
                   Academy.Instance.IsCommunicatorOn;
        }
    }

    bool RecordingActive {
        get {
            return demonstrationRecorder != null &&
                   demonstrationRecorder.Record;
        }
    }

    public void Configure(
        PlayerControl control,
        Rigidbody2D bodyRigidbody,
        Rigidbody2D hammerRigidbody,
        Camera camera,
        RLWaypoint[] routeWaypoints = null
    ) {
        playerControl = control;
        body = bodyRigidbody;
        hammer = hammerRigidbody;
        followCamera = camera;
        if (routeWaypoints != null) {
            waypoints = routeWaypoints;
        }
    }

    public override void Initialize() {
        ResolveReferences();
        CacheInitialState();
        playerControl.SetExternalControlEnabled(CommunicatorActive);
    }

    void ResolveReferences() {
        if (playerControl == null) {
            playerControl = GetComponent<PlayerControl>();
        }

        if (body == null && playerControl != null &&
            playerControl.body != null) {
            body = playerControl.body.GetComponent<Rigidbody2D>();
        }

        if (hammer == null && playerControl != null &&
            playerControl.hammerHead != null) {
            hammer =
                playerControl.hammerHead.GetComponent<Rigidbody2D>();
        }

        if (followCamera == null) {
            followCamera = Camera.main;
        }

        if (demonstrationRecorder == null) {
            demonstrationRecorder =
                GetComponent<DemonstrationRecorder>();
        }

        if (playerControl == null || body == null || hammer == null) {
            throw new MissingReferenceException(
                "GettingOverItAgent requires PlayerControl, body, and " +
                "hammer Rigidbody2D references.");
        }

        hammerCollider = hammer.GetComponent<Collider2D>();
    }

    void CacheInitialState() {
        initialBodyPosition = body.position;
        initialBodyRotation = body.rotation;
        initialHammerPosition = hammer.position;
        initialHammerRotation = hammer.rotation;
        initialCameraPosition =
            followCamera == null ? Vector3.zero :
            followCamera.transform.position;
        maximumHeight = body.position.y;
        initialStateCached = true;
    }

    public override void OnEpisodeBegin() {
        ResolveReferences();
        if (!initialStateCached) {
            CacheInitialState();
        }

        body.position = initialBodyPosition;
        body.rotation = initialBodyRotation;
        body.velocity = Vector2.zero;
        body.angularVelocity = 0.0f;

        hammer.position = initialHammerPosition;
        hammer.rotation = initialHammerRotation;
        hammer.velocity = Vector2.zero;
        hammer.angularVelocity = 0.0f;

        if (followCamera != null) {
            followCamera.transform.position = initialCameraPosition;
        }

        previousAction = Vector2.zero;
        maximumHeight = initialBodyPosition.y;
        ResetWaypointProgress();
        playerControl.ResetControlState();
        playerControl.SetExternalControlEnabled(CommunicatorActive);
        Physics2D.SyncTransforms();
    }

    void FixedUpdate() {
        bool communicatorActive = CommunicatorActive;
        bool recordingActive = RecordingActive;
        if (communicatorActive && recordingActive) {
            throw new System.InvalidOperationException(
                "Human demonstration recording cannot run while the " +
                "Python communicator is connected. Disable recording " +
                "through RL > Demonstrations > Disable Recording.");
        }
        playerControl.SetExternalControlEnabled(communicatorActive);

        if (!communicatorActive && !recordingActive) {
            return;
        }

        ValidateWaypointConfiguration();

        float height = body.position.y;
        if (height > maximumHeight) {
            AddReward(
                (height - maximumHeight) * heightProgressScale);
            maximumHeight = height;
        }

        AddReward(stepPenalty);

        if (UpdateWaypointProgress()) {
            return;
        }

        if (height >= goalY) {
            AddReward(successReward);
            EndEpisode();
        } else if (height <= fallY) {
            AddReward(fallPenalty);
            EndEpisode();
        }
    }

    void ValidateWaypointConfiguration() {
        if (waypointConfigurationValidated) {
            return;
        }
        if (waypoints == null || waypoints.Length < 2) {
            throw new MissingReferenceException(
                "Waypoint curriculum requires at least two ordered " +
                "waypoints. Run RL > Configure Main Scene.");
        }
        for (int i = 0; i < waypoints.Length; i++) {
            if (waypoints[i] == null) {
                throw new MissingReferenceException(
                    "Waypoint curriculum contains an unassigned marker " +
                    "at index " + i + ".");
            }
        }
        waypointConfigurationValidated = true;
    }

    void ResetWaypointProgress() {
        activeWaypointIndex = 0;
        previousWaypointDistance = HasActiveWaypoint()
            ? DistanceToActiveWaypoint()
            : 0.0f;
    }

    bool UpdateWaypointProgress() {
        if (!HasActiveWaypoint()) {
            return false;
        }

        RLWaypoint waypoint = waypoints[activeWaypointIndex];
        float distance = DistanceToActiveWaypoint();
        AddReward(
            (previousWaypointDistance - distance) *
            waypointProgressScale);
        previousWaypointDistance = distance;

        if (!waypoint.Contains(body.position)) {
            return false;
        }

        bool finalWaypoint =
            activeWaypointIndex == waypoints.Length - 1;
        AddReward(finalWaypoint
            ? finalWaypointReward
            : waypointReward);
        activeWaypointIndex++;

        if (finalWaypoint && terminateAtFinalWaypoint) {
            EndEpisode();
            return true;
        }

        previousWaypointDistance = HasActiveWaypoint()
            ? DistanceToActiveWaypoint()
            : 0.0f;
        return false;
    }

    bool HasActiveWaypoint() {
        return waypoints != null &&
               activeWaypointIndex >= 0 &&
               activeWaypointIndex < waypoints.Length &&
               waypoints[activeWaypointIndex] != null;
    }

    float DistanceToActiveWaypoint() {
        return Vector2.Distance(
            body.position,
            waypoints[activeWaypointIndex].transform.position);
    }

    public override void CollectObservations(VectorSensor sensor) {
        Vector2 bodyRelativePosition =
            body.position - initialBodyPosition;
        Vector2 hammerRelativePosition =
            hammer.position - body.position;
        float safeMaxRange =
            Mathf.Max(playerControl.maxRange, Mathf.Epsilon);

        sensor.AddObservation(Clip(bodyRelativePosition / positionScale));
        sensor.AddObservation(Clip(body.velocity / bodyVelocityScale));
        sensor.AddObservation(Clip(
            hammerRelativePosition / safeMaxRange));
        sensor.AddObservation(Clip(
            hammer.velocity / hammerVelocityScale));
        sensor.AddObservation(
            playerControl.HammerTouchingTerrain ? 1.0f : 0.0f);
        sensor.AddObservation(Mathf.Clamp(
            (maximumHeight - initialBodyPosition.y) / heightScale,
            -1.0f,
            1.0f));
        sensor.AddObservation(previousAction);

        for (int i = 0; i < 8; i++) {
            float angle = i * 45.0f * Mathf.Deg2Rad;
            Vector2 direction =
                new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));
            sensor.AddObservation(GetTerrainProximity(direction));
        }
    }

    float GetTerrainProximity(Vector2 direction) {
        float hammerRadius = 0.0f;
        if (hammerCollider != null) {
            Vector2 boundsSize = hammerCollider.bounds.size;
            hammerRadius = Mathf.Min(boundsSize.x, boundsSize.y) * 0.5f;
        }

        float castDistance = rayLength + hammerRadius;
        int hitCount = Physics2D.RaycastNonAlloc(
            hammer.position,
            direction,
            raycastResults,
            castDistance,
            terrainMask);

        float closestDistance = float.PositiveInfinity;
        for (int i = 0; i < hitCount; i++) {
            RaycastHit2D hit = raycastResults[i];
            if (hit.collider == null ||
                hit.collider.transform.IsChildOf(transform)) {
                continue;
            }
            closestDistance = Mathf.Min(closestDistance, hit.distance);
        }

        if (float.IsPositiveInfinity(closestDistance)) {
            return 0.0f;
        }

        float clearance = Mathf.Max(
            0.0f, closestDistance - hammerRadius);
        return 1.0f - Mathf.Clamp01(clearance / rayLength);
    }

    public override void OnActionReceived(ActionBuffers actions) {
        bool communicatorActive = CommunicatorActive;
        if (!communicatorActive && !RecordingActive) {
            return;
        }

        Vector2 action = new Vector2(
            actions.ContinuousActions[0],
            actions.ContinuousActions[1]);
        action = Vector2.ClampMagnitude(action, 1.0f);

        previousAction = action;
        if (communicatorActive) {
            playerControl.SetExternalCommand(action);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut) {
        Vector2 mouseCommand = playerControl.GetMouseCommand();
        ActionSegment<float> continuousActions =
            actionsOut.ContinuousActions;
        continuousActions[0] = mouseCommand.x;
        continuousActions[1] = mouseCommand.y;
    }

    protected override void OnDisable() {
        if (playerControl != null) {
            playerControl.SetExternalControlEnabled(false);
        }
        base.OnDisable();
    }

    static Vector2 Clip(Vector2 value) {
        return new Vector2(
            Mathf.Clamp(value.x, -1.0f, 1.0f),
            Mathf.Clamp(value.y, -1.0f, 1.0f));
    }
}
