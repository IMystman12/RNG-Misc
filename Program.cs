using System.Diagnostics;

Console.WriteLine("Hello, World!");

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
    Console.WriteLine($"{graph.points.Count} {funPredicate.w} {funPredicate.b} {sw.ElapsedMilliseconds}");
    d += Random.Shared.NextDouble();
}
sw.Stop();

public struct LinearFunction
{
    public double w, b;
    public double Calculate(double x) => w * x + b;
    public static bool operator ==(LinearFunction a, LinearFunction b) => a.w == b.w && a.b == b.b;
    public static bool operator !=(LinearFunction a, LinearFunction b) => !(a == b);
    public static LinearFunction random => new()
    {
        w = Random.Shared.NextDouble(),
        b = Random.Shared.NextDouble()
    };
    static LinearFunction temp = new();
    public void Train(Graph graph)
    {
        double sx = 0, sy = 0, sxy = 0, sx2 = 0, n;
        if (graph.points.Count > 1)
        {
            n = graph.points.Count;
            graph.points.ForEach(a =>
            {
                sx += a.x;
                sy += a.y;
                sxy += a.x * a.y;
                sx2 += a.x * a.x;
            });
            w = (n * sxy - sx * sy) / (n * sx2 - sx * sx);
            b = (sy - w * sx) / n;
        }
    }
}
public struct Graph
{
    public List<Point> points;
    public Graph() => points = new List<Point>();
}
public struct Point
{
    public double x, y;
    public bool delta;
    public static Point operator -(Point a, Point b) => new Point()
    {
        x = a.x - b.x,
        y = a.y - b.y,
        delta = !a.delta && !b.delta
    };
    public double k => y / x;
    public static Point random => new()
    {
        x = Random.Shared.NextDouble(),
        y = Random.Shared.NextDouble()
    };
    public override string ToString() => $"{x},{y}";
}