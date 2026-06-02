using System.Diagnostics;

Console.WriteLine("Hello, World!");

Graph graph = new();
NonlinearFunctionTorch nonlinearFunction = new(99);
double d = 0;

Stopwatch sw = Stopwatch.StartNew();
while (true)
{
    d = Random.Shared.NextDouble() + 99;
    graph.points.Add(new Point() { x = d, y = TestFunc(d) });
    while (graph.points.Count > 1024)
    {
        graph.points.RemoveAt(0);
    }
    if (sw.ElapsedMilliseconds % 1024 == 0)
    {
        for (int i = 0; i < 99; i++)
        {
            nonlinearFunction.Train(graph);
        }
        double r = TestFunc(d) - nonlinearFunction.Calculate(d);
        Console.WriteLine($"{TestFunc(d)}-{nonlinearFunction.Calculate(d)}={r}");
        graph.points.Clear();
    }
}
sw.Stop();

double TestFunc(double x)
{
    return x*99;
}
