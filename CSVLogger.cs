
using System.IO;
using UnityEngine;

public class CSVLogger : MonoBehaviour
{
    public string fileName = "unity_generations_log.csv";
    StreamWriter sw;

    void Awake()
    {
        var path = Path.Combine(Application.persistentDataPath, fileName);
        sw = new StreamWriter(path, false);
        sw.WriteLine("generation,best_fitness,mean_fitness,best_dx");
        sw.Flush();
        Debug.Log("CSV en: " + path);
    }

    public void LogGen(int gen, float best, float mean, float dx)
    {
        sw.WriteLine($"{gen},{best:F6},{mean:F6},{dx:F6}");
        sw.Flush();
    }

    void OnDestroy(){ if (sw!=null) sw.Close(); }
}
