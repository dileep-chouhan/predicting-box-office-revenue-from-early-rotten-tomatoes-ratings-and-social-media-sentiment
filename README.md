# Predicting Box Office Revenue from Early Rotten Tomatoes Ratings and Social Media Sentiment

**Overview:**

This project aims to build a predictive model for first-week box office revenue of movies.  The model leverages early Rotten Tomatoes critic scores and aggregated social media sentiment (obtained from a hypothetical source, not included in this repository for simplicity) as key predictors.  The analysis explores the relationship between these factors and box office performance, ultimately aiming to assist film studios in optimizing their marketing budget allocation based on early indicators of success.  The project involves data cleaning, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

**Technologies Used:**

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn (for model building)


**How to Run:**

1. **Clone the repository:**  `git clone <repository_url>`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Run the main script:** `python main.py`

**Example Output:**

The script will print key analysis results to the console, including details about the chosen model, its performance metrics (e.g., R-squared, RMSE), and the feature importance.  Additionally, the script will generate several visualization files (e.g., scatter plots showing the relationship between Rotten Tomatoes scores and box office revenue, and bar charts visualizing feature importances), saved in the `output` directory.  These visualizations provide insights into the relationships between the predictive variables and the target variable (box office revenue).  The exact filenames of the generated plots may vary.


**Data:**

Note that this repository does *not* include the raw movie data used for this project. This is due to limitations in publicly available datasets that contain all the necessary information. To reproduce the results, you would need to provide your own dataset with columns representing Rotten Tomatoes scores, social media sentiment, and first-week box office revenue.


**Future Work:**

Future improvements could include incorporating additional features (e.g., genre, actors, budget), exploring more sophisticated modeling techniques, and integrating a more robust social media sentiment analysis pipeline.  Further research could also focus on improving the accuracy of the model and expanding its applicability across different film genres and market segments.