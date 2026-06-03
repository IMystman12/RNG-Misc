using System.Diagnostics;
var sw = Stopwatch.StartNew();
int inputCount = 9, trainSamples = 1024, c = 0;
MultiInputNonlinearFunction torchFunction = new(inputCount, 9, 9);
double[,] transSets = new double[trainSamples, inputCount];
double[] answers = new double[trainSamples];
for (int i = 0; i < trainSamples; i++)
{
    for (int j = 0; j < inputCount; j++)
    {
        transSets[i, j] = rng();
    }
    answers[i] = rng();
}
Console.WriteLine($"{sw.ElapsedMilliseconds} 1");
for (int i = 0; i < 1024; i++)
{
    torchFunction.Train(transSets, answers);
}
Console.WriteLine($"{sw.ElapsedMilliseconds} 2");
double[] testSets = new double[inputCount];
for (int j = 0; j < inputCount; j++)
{
    testSets[j] = rng();
}
double a = rng(), b = torchFunction.Calculate(testSets);
Console.WriteLine($"{a} {b} {a - b}");

double rng()
{
    c++;
    return Math.Sin(c);
}