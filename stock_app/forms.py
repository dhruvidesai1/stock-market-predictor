from django import forms

MODEL_CHOICES = [
    ('mlp', 'MLP (GA)'),
    ('lstm', 'LSTM (GA)'),
    ('hybrid_mlp', 'MLP (GA then Backprop)'),
    ('hybrid_lstm', 'LSTM (GA then Backprop)'),
]

class PredictForm(forms.Form):
    ticker = forms.CharField(max_length=100, initial='AAPL', label="Ticker (comma separated for multiple)")
    start_date = forms.DateField(widget=forms.DateInput(attrs={'type':'date'}))
    end_date = forms.DateField(widget=forms.DateInput(attrs={'type':'date'}))
    window = forms.IntegerField(min_value=5, max_value=60, initial=20)
    model_type = forms.ChoiceField(choices=MODEL_CHOICES)
    ga_population = forms.IntegerField(min_value=4, max_value=100, initial=12)
    ga_ngen = forms.IntegerField(min_value=1, max_value=50, initial=6)
    fine_tune_epochs = forms.IntegerField(min_value=0, max_value=50, initial=5, required=False)
