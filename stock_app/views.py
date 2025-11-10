from django.shortcuts import render
from .forms import PredictForm
from .ga_ann_model import predict_with_ga_ann

def index(request):
    results = []
    form = PredictForm(request.POST or None)
    error = None
    if request.method == 'POST' and form.is_valid():
        tickers_raw = form.cleaned_data['ticker']
        tickers = [t.strip().upper() for t in tickers_raw.split(',') if t.strip()]
        start_date = str(form.cleaned_data['start_date'])
        end_date = str(form.cleaned_data['end_date'])
        window = form.cleaned_data['window']
        model_choice = form.cleaned_data['model_type']
        ga_pop = form.cleaned_data['ga_population']
        ga_ngen = form.cleaned_data['ga_ngen']
        fine_tune_epochs = form.cleaned_data.get('fine_tune_epochs') or 0

        for t in tickers:
            try:
                fine_tune = model_choice.startswith('hybrid')
                model_type = 'lstm' if 'lstm' in model_choice else 'mlp'
                res = predict_with_ga_ann(
                    ticker=t,
                    start_date=start_date,
                    end_date=end_date,
                    model_type=model_type,
                    window=window,
                    ga_population=ga_pop,
                    ga_ngen=ga_ngen,
                    fine_tune=fine_tune,
                    fine_tune_epochs=fine_tune_epochs
                )
                results.append(res)
            except Exception as e:
                error = str(e)
                break

    return render(request, 'stock_app/index.html', {'form': form, 'results': results, 'error': error})
