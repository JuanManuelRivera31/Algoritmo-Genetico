
using UnityEngine;

public class CreatureController : MonoBehaviour
{
    public CreatureDNA dna;
    public Rigidbody[] segments;
    public ConfigurableJoint[] joints;
    public PhysicsMaterial footMaterial;
    public float episodeTime = 12f;
    public Transform startMarker;

    private float t;
    private Vector3 startPos;
    private float energyJoules;
    private float slipMeters;
    private bool fell;
    private float heightRef;

    void Start()
    {
        if (startMarker) transform.position = startMarker.position;
        startPos = transform.position;

        float scale = dna.altura / 1.7f;
        transform.localScale = new Vector3(scale, scale, scale);
        foreach (var rb in segments) if(rb) rb.mass *= dna.masa / 70f;

        if (footMaterial) { footMaterial.dynamicFriction = dna.fricPie; footMaterial.staticFriction = dna.fricPie; }

        for (int i=0;i<joints.Length;i++){
            var jd = joints[i];
            jd.angularXDrive = new JointDrive {
                positionSpring = dna.kLig[i],
                positionDamper = dna.cLig[i],
                maximumForce   = dna.tauMax[i] * Mathf.Pow(dna.masa/70f, 2f/3f)
            };
            joints[i] = jd;
        }
        heightRef = dna.altura;
    }

    void FixedUpdate()
    {
        t += Time.fixedDeltaTime;

        for (int i=0;i<joints.Length;i++){
            float targetRad = dna.offset[i] + dna.amp[i] * Mathf.Sin(2f*Mathf.PI * dna.freq * t + dna.fase[i]);
            var target = Quaternion.Euler(Mathf.Rad2Deg*targetRad, 0f, 0f);
            var jd = joints[i];
            jd.targetRotation = Quaternion.Inverse(target);
            joints[i] = jd;

            float spring = dna.kLig[i];
            float tau = Mathf.Min(dna.tauMax[i], spring * Mathf.Abs(targetRad)) * Mathf.Pow(dna.masa/70f, 2f/3f);
            float qdot = Mathf.Abs(joints[i].targetAngularVelocity.x);
            energyJoules += tau * qdot * Time.fixedDeltaTime;
        }

        fell = fell || FellCondition();

        if (t >= episodeTime) enabled = false;
    }

    bool FellCondition()
    {
        var com = GetCenterOfMass();
        bool low = com.y < 0.35f * heightRef;
        float pitch = transform.localEulerAngles.z;
        if (pitch > 180f) pitch -= 360f;
        bool tilt = Mathf.Abs(pitch) > 35f;
        return low || tilt;
    }

    Vector3 GetCenterOfMass()
    {
        Vector3 s = Vector3.zero; float m = 0f;
        foreach (var rb in segments){ if(!rb) continue; s += rb.worldCenterOfMass * rb.mass; m += rb.mass; }
        return s / Mathf.Max(0.0001f, m);
    }

    public (float fitness, float dx, bool fell) Evaluate()
    {
        float dx = transform.position.x - startPos.x;
        float alpha=0.3f, beta=0.2f, gamma=2.0f, delta=0.1f;
        float ePerKgM = (dx>0.5f) ? (energyJoules/(dna.masa*dx)) : energyJoules/(dna.masa*0.5f);
        float fitness = dx - alpha*ePerKgM - beta*0f - delta*slipMeters - (fell?gamma:0f);
        return (fitness, dx, fell);
    }
}
