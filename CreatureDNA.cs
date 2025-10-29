
using UnityEngine;
using System;

[Serializable]
public class CreatureDNA
{
    public static readonly Vector2 ALTURA_R = new(1.3f, 2.1f);
    public static readonly Vector2 MASA_R   = new(45f, 95f);
    public static readonly Vector2 FRIC_R   = new(0.6f, 1.2f);
    public static readonly Vector2 FREQ_R   = new(0.6f, 3.0f);
    public static readonly Vector2 AMP_R    = new(Mathf.Deg2Rad * 3f, Mathf.Deg2Rad * 20f);
    public static readonly Vector2 OFF_R    = new(Mathf.Deg2Rad * -10f, Mathf.Deg2Rad * 10f);
    public static readonly Vector2 FASE_R   = new(0f, Mathf.PI * 2f);
    public static readonly Vector2 TORQUE_R = new(80f, 220f);
    public static readonly Vector2 KLIG_R   = new(1000f, 6000f);
    public static readonly Vector2 CLIG_R   = new(5f, 60f);

    public float altura;
    public float masa;
    public float fricPie;
    public float freq;
    public float[] amp = new float[6];
    public float[] fase = new float[6];
    public float[] offset = new float[6];
    public float[] tauMax = new float[6];
    public float[] kLig = new float[6];
    public float[] cLig = new float[6];

    public int JointCount => 6;

    public static CreatureDNA Random(System.Random rng)
    {
        CreatureDNA d = new CreatureDNA();
        d.altura = Rand(ALTURA_R, rng);
        d.masa   = Rand(MASA_R, rng);
        d.fricPie= Rand(FRIC_R, rng);
        d.freq   = Rand(FREQ_R, rng);
        for (int i=0;i<6;i++){
            d.amp[i]    = Rand(AMP_R, rng);
            d.fase[i]   = Rand(FASE_R, rng);
            d.offset[i] = Rand(OFF_R, rng);
            d.tauMax[i] = Rand(TORQUE_R, rng);
            d.kLig[i]   = Rand(KLIG_R, rng);
            d.cLig[i]   = Rand(CLIG_R, rng);
        }
        return d;
    }

    public CreatureDNA Clone() => JsonUtility.FromJson<CreatureDNA>(JsonUtility.ToJson(this));

    public void Mutate(System.Random rng, float p=0.15f, float sigmaRel=0.05f)
    {
        void Mut(ref float v, Vector2 r)
        {
            if (rng.NextDouble() < p)
            {
                float span = r.y - r.x;
                v += (float)NextGaussian(rng) * sigmaRel * span;
                v = Mathf.Clamp(v, r.x, r.y);
            }
        }
        Mut(ref altura, ALTURA_R);
        Mut(ref masa, MASA_R);
        Mut(ref fricPie, FRIC_R);
        Mut(ref freq, FREQ_R);

        for (int i=0;i<6;i++){
            amp[i]    = ClampMut(amp[i], AMP_R, p, sigmaRel, rng);
            fase[i]   = ClampMut(fase[i], FASE_R, p, sigmaRel, rng);
            offset[i] = ClampMut(offset[i], OFF_R, p, sigmaRel, rng);
            tauMax[i] = ClampMut(tauMax[i], TORQUE_R, p, sigmaRel, rng);
            kLig[i]   = ClampMut(kLig[i], KLIG_R, p, sigmaRel, rng);
            cLig[i]   = ClampMut(cLig[i], CLIG_R, p, sigmaRel, rng);
        }
    }

    public static CreatureDNA CrossoverUniform(CreatureDNA a, CreatureDNA b, System.Random rng)
    {
        var c = a.Clone();
        c.altura  = rng.NextDouble()<0.5?a.altura:b.altura;
        c.masa    = rng.NextDouble()<0.5?a.masa:b.masa;
        c.fricPie = rng.NextDouble()<0.5?a.fricPie:b.fricPie;
        c.freq    = 0.5f*(a.freq+b.freq);
        for(int i=0;i<6;i++){
            bool sel = rng.NextDouble()<0.5;
            c.amp[i]    = sel?a.amp[i]:b.amp[i];
            c.fase[i]   = sel?a.fase[i]:b.fase[i];
            c.offset[i] = sel?a.offset[i]:b.offset[i];
            c.tauMax[i] = Mathf.Min(a.tauMax[i], b.tauMax[i]);
            c.kLig[i]   = sel?a.kLig[i]:b.kLig[i];
            c.cLig[i]   = sel?a.cLig[i]:b.cLig[i];
        }
        return c;
    }

    static float Rand(Vector2 r, System.Random rng) => (float)(r.x + (r.y - r.x) * rng.NextDouble());
    static float ClampMut(float v, Vector2 r, float p, float s, System.Random rng)
    {
        if (rng.NextDouble() < p)
        {
            var span = r.y - r.x;
            v += (float)NextGaussian(rng) * s * span;
            v = Mathf.Clamp(v, r.x, r.y);
        }
        return v;
    }
    public static double NextGaussian(System.Random rng)
    {
        var u1 = 1.0 - rng.NextDouble();
        var u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
