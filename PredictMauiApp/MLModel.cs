
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;

public class MLModel
{
    private readonly PredictionEngine<ModelInput, ModelOutput> _predictor;

    public MLModel()
    {
        var ml = new MLContext();

        var data = new List<ModelInput>()
        {
            new ModelInput { X = 1, Y = 2 },
            new ModelInput { X = 2, Y = 4 },
            new ModelInput { X = 3, Y = 6 },
            new ModelInput { X = 4, Y = 8 },
        };

        var trainData = ml.Data.LoadFromEnumerable(data);

        var pipeline = ml.Transforms
            .Concatenate("Features", nameof(ModelInput.X))
            .Append(ml.Regression.Trainers.Sdca(labelColumnName: "Y"));

        var model = pipeline.Fit(trainData);

        _predictor = ml.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
    }

    public float Predict(float x)
    {
        var result = _predictor.Predict(new ModelInput { X = x });
        return result.PredictedY;
    }
}

public class ModelInput
{
    public float X { get; set; }
    public float Y { get; set; }
}

public class ModelOutput
{
    [ColumnName("Score")]
    public float PredictedY { get; set; }
}