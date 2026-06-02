using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

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
    public Tensor tensorX => ToVaildTensor(points.Select(a => a.x).ToArray());
    public Tensor tensorY => ToVaildTensor(points.Select(a => a.y).ToArray());
    public static Tensor ToVaildTensor(params double[] array)
    {
        var result = new double[array.Length, 1];
        for (int i = 0; i < array.Length; i++)
        {
            result[i, 0] = array[i];
        }
        return tensor(result, ScalarType.Float32, NonlinearFunctionTorch.device);
    }
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
[Obsolete("Invaild!", true)]
public struct NonlinearFunction
{
    public int count;
    public double[] linearFunctionWs;
    public LinearFunction[] linearFunctions;
    public double b, learningRate = 0.001d;
    public NonlinearFunction(int count)
    {
        this.count = count;
        linearFunctionWs = new double[count];
        linearFunctions = new LinearFunction[count];
    }
    public double Calculate(double x)
    {
        double result = 0;
        for (int i = 0; i < count; i++)
        {
            result += linearFunctionWs[i] * Math.Max(0, linearFunctions[i].Calculate(x));
        }
        return result + b;
    }
    public void Train(Graph graph)
    {
        foreach (var a in graph.points)
        {
            double d = 0;
            double[] preAns = new double[count];
            for (int i = 0; i < count; i++)
            {
                preAns[i] = Math.Max(0, linearFunctions[i].Calculate(a.x));
                d += linearFunctionWs[i] * preAns[i];
            }
            d = (a.y - d) * learningRate;

            b += d;
            for (int i = 0; i < count; i++)
            {
                linearFunctionWs[i] += d;
                if (preAns[i] > 0)
                {
                    linearFunctions[i].w += d;
                    linearFunctions[i].b += d;
                }
            }
        }
    }
}

public struct NonlinearFunctionTorch
{
    public static Device device = new Device(DeviceType.CPU);
    Sequential model;
    Adam optimizer;
    public NonlinearFunctionTorch(int count)
    {
        model = nn.Sequential(
            nn.Linear(1, count),
            nn.ReLU(),
            nn.Linear(count, 1)
            //nn.Flatten()
            );
        optimizer = optim.Adam(model.parameters(), 0.01f);
    }
    public double Calculate(double x) => (double)model.forward(Graph.ToVaildTensor(x));
    public void Train(Graph graph)
    {
        var loss = nn.functional.mse_loss(model.forward(graph.tensorX), graph.tensorY);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}
