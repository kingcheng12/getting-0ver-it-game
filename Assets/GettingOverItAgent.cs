using Unity.MLAgents;
using Unity.MLAgents.Actuators;
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

    [Header("Observation normalization")]
    [SerializeField] float positionScale = 10.0f;
    [SerializeField] float bodyVelocityScale = 6.0f;
    [SerializeField] float hammerVelocityScale = 10.0f;
    [SerializeField] float heightScale = 6.5f;
    [SerializeField] float rayLength = 2.0f;
    [SerializeField] LayerMask terrainMask = 1;

    [Header("Episode")]
    [SerializeField] float goalY = 6.5f;
    [SerializeField] float fallY = -3.0f;
    [SerializeField] float stepPenalty = -0.0001f;
    [SerializeField] float successReward = 10.0f;
    [SerializeField] float fallPenalty = -1.0f;

    Vector2 initialBodyPosition;
    float initialBodyRotation;
    Vector2 initialHammerPosition;
    float initialHammerRotation;
    Vector3 initialCameraPosition;
    Vector2 previousAction;
    float maximumHeight;
    bool initialStateCached;
    Collider2D hammerCollider;
    readonly RaycastHit2D[] raycastResults =
        new RaycastHit2D[RaycastBufferSize];

    bool CommunicatorActive {
        get {
            return Academy.IsInitialized &&
                   Academy.Instance.IsCommunicatorOn;
        }
    }

    public void Configure(
        PlayerControl control,
        Rigidbody2D bodyRigidbody,
        Rigidbody2D hammerRigidbody,
        Camera camera
    ) {
        playerControl = control;
        body = bodyRigidbody;
        hammer = hammerRigidbody;
        followCamera = camera;
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
        playerControl.ResetControlState();
        playerControl.SetExternalControlEnabled(CommunicatorActive);
        Physics2D.SyncTransforms();
    }

    void FixedUpdate() {
        bool communicatorActive = CommunicatorActive;
        playerControl.SetExternalControlEnabled(communicatorActive);

        if (!communicatorActive) {
            return;
        }

        float height = body.position.y;
        if (height > maximumHeight) {
            AddReward(height - maximumHeight);
            maximumHeight = height;
        }

        AddReward(stepPenalty);

        if (height >= goalY) {
            AddReward(successReward);
            EndEpisode();
        } else if (height <= fallY) {
            AddReward(fallPenalty);
            EndEpisode();
        }
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
        if (!CommunicatorActive) {
            return;
        }

        Vector2 action = new Vector2(
            actions.ContinuousActions[0],
            actions.ContinuousActions[1]);
        action = Vector2.ClampMagnitude(action, 1.0f);

        previousAction = action;
        playerControl.SetExternalCommand(action);
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
