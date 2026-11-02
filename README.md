# Betting Model Overfitting Bookmaker Margin

This example simulates a simple betting model that appears successful on a small, biased training dataset due to overfitting to a spurious correlation. When evaluated on a larger, truly random test dataset, the model fails to generalize and results in a significant loss, primarily due to its inability to overcome the inherent bookmaker margin. It illustrates why models that look good on historical data often fail in real-world betting scenarios.

## Language

`python`

## How to Run

Save the code as `main.py`.
Run from your terminal: `python main.py`

## Original Article

This example accompanies the Turkish article: [Bahis Modelleri Neden Başarısız Olur? 3.671 Tahminlik Bir Vaka Analizi](https://fatihsoysal.com/blog/bahis-modelleri-neden-basarisiz-olur-3-671-tahminlik-bir-vaka-analizi/).

## License

MIT — see [LICENSE](LICENSE).
