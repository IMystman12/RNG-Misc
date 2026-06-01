using System.Diagnostics;

Console.WriteLine("Hello, World!");

Graph graph = new();
NonlinearFunction nonlinearFunction = new(9);
double d = 0;

Stopwatch sw = Stopwatch.StartNew();
while (true)
{
    graph.points.Add(new Point() { x = d, y = Math.Sin(d) });
    nonlinearFunction.Train(graph);
    d += Random.Shared.NextDouble();
    if (sw.ElapsedMilliseconds % 1024 == 0)
    {
        double r = Math.Sin(d) - nonlinearFunction.Calculate(d);
        Console.WriteLine($"{Math.Sin(d)}-{nonlinearFunction.Calculate(d)}={r}");
    }
}
sw.Stop();
