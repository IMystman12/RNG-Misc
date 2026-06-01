using System.Diagnostics;
public static class LinearTest
{
    public static void Main()
    {
        Console.WriteLine("Hello, World! Linear");

        Graph graph = new();
        LinearFunction funPredicate = LinearFunction.random, funCal = new()
        {
            w = double.Parse(Console.ReadLine()),
            b = double.Parse(Console.ReadLine())
        };
        double d = 0;

        Stopwatch sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < 1024)
        {
            graph.points.Add(new Point() { x = d, y = funCal.Calculate(d) });
            funPredicate.Train(graph);
            d += Random.Shared.NextDouble();
        }
        sw.Stop();
        Console.WriteLine($"{graph.points.Count} {funPredicate.w} {funPredicate.b} {sw.ElapsedMilliseconds}");
    }
}