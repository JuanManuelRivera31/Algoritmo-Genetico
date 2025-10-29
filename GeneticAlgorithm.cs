
using UnityEngine;
using System.Collections.Generic;

public class GeneticAlgorithm : MonoBehaviour
{
    public int populationSize = 40;
    public int generations = 30;
    public int elitism = 3;
    public int tournamentK = 3;
    public float mutationProb = 0.15f;
    public float mutationSigmaRel = 0.05f;
    public int seed = 1234;
    public float episodeTime = 12f;

    public GameObject creaturePrefab;
    public Transform spawnPoint;
    public int lanes = 4;
    public float laneSpacing = 2.0f;
    public CSVLogger logger;

    System.Random rng;
    List<CreatureDNA> population;

    void Start()
    {
        rng = new System.Random(seed);
        population = new List<CreatureDNA>();
        for(int i=0;i<populationSize;i++) population.Add(CreatureDNA.Random(rng));
        StartCoroutine(Evolve());
    }

    System.Collections.IEnumerator Evolve()
    {
        float bestSoFar = float.NegativeInfinity;
        int bad = 0, earlyPatience = 8;
        float minDelta = 0.01f;

        for (int g=0; g<generations; g++)
        {
            List<(CreatureDNA dna, float fit, float dx)> scored = new();

            for (int i=0; i<population.Count; i+=lanes)
            {
                var batch = new List<CreatureController>();
                for (int b=0; b<lanes && (i+b)<population.Count; b++)
                {
                    var pos = spawnPoint.position + new Vector3(0f, 0f, b*laneSpacing);
                    var go = Instantiate(creaturePrefab, pos, Quaternion.identity);
                    var ctrl = go.GetComponent<CreatureController>();
                    ctrl.dna = population[i+b].Clone();
                    ctrl.episodeTime = episodeTime;
                    batch.Add(ctrl);
                }

                float t=0f;
                while (t < episodeTime)
                {
                    t += Time.deltaTime;
                    yield return null;
                }

                foreach (var ctrl in batch)
                {
                    var (fit, dx, fell) = ctrl.Evaluate();
                    scored.Add((ctrl.dna, fit, dx));
                    Destroy(ctrl.gameObject);
                }
                yield return null;
            }

            scored.Sort((a,b)=> b.fit.CompareTo(a.fit));
            float mean=0f; foreach(var s in scored) mean += s.fit; mean/=scored.Count;
            Debug.Log($"Gen {g} | Best {scored[0].fit:F3} | Mean {mean:F3} | Dx {scored[0].dx:F2}");
            if (logger) logger.LogGen(g, scored[0].fit, mean, scored[0].dx);

            if (scored[0].fit > bestSoFar + minDelta) { bestSoFar = scored[0].fit; bad = 0; }
            else { bad++; if (bad>=earlyPatience){ Debug.Log($"[EarlyStop] stop at gen {g}"); yield break; } }

            List<CreatureDNA> next = new();
            for (int e=0; e<elitism; e++) next.Add(scored[e].dna.Clone());
            while (next.Count < populationSize)
            {
                var p1 = Tournament(scored, tournamentK);
                var p2 = Tournament(scored, tournamentK);
                var child = CreatureDNA.CrossoverUniform(p1, p2, rng);
                child.Mutate(rng, mutationProb, mutationSigmaRel);
                next.Add(child);
            }
            population = next;
        }
    }

    CreatureDNA Tournament(List<(CreatureDNA dna, float fit, float dx)> scored, int k)
    {
        (CreatureDNA dna, float fit, float dx) best = (null, float.NegativeInfinity, 0f);
        for (int i=0;i<k;i++){
            var pick = scored[rng.Next(scored.Count)];
            if (pick.fit > best.fit) best = pick;
        }
        return best.dna.Clone();
    }
}
