namespace PredictMauiApp
{
    public partial class MainPage : ContentPage
    {
        private readonly MLModel _model;

        public MainPage()
        {
            InitializeComponent();
            _model = new MLModel();
        }

        private void OnPredictClicked(object sender, EventArgs e)
        {
            if (float.TryParse(InputBox.Text, out float x))
            {
                float prediction = _model.Predict(x);

                ResultLabel.Text = $"Prediction: {prediction}";
            }
            else
            {
                ResultLabel.Text = "Please enter a valid number";
            }
        }


    }
}
