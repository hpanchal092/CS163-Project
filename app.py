from dash import Dash, html

app = Dash(__name__)
server = app.server

def fig(filename, caption):
    return html.Div([
        html.Img(src=app.get_asset_url(filename), className="figure-img"),
        html.P(caption, className="caption")
    ], className="figure-card")

app.layout = html.Div([

    html.H1("Early Prediction of E-commerce Product Success"),
    html.H3("CS 163 Data Science Senior Project – Group 13"),
    html.P("Members: Vince Lai, Harsh Panchal"),

    html.H2("Executive Summary"),
    html.P("""
    This project predicts whether an e-commerce product will become successful using only
    the first 4 weeks of post-launch review data. We define success as products in the top
    10% of their category based on lifetime review count. We compare Logistic Regression
    and XGBoost using early review features such as ratings, votes, review count, and
    review velocity.
    """),

    html.H2("Project Objective"),
    html.P("""
    The goal is to help sellers and marketplace platforms identify promising products early.
    Instead of waiting months to evaluate product performance, our model uses early review
    behavior to rank products by success potential.
    """),

    html.H2("Data Sources"),
    html.Ul([
        html.Li("Amazon Customer Reviews Dataset from UCSD / McAuley Lab"),
        html.Li("5-core product category subsets"),
        html.Li("Categories: Electronics, Clothing, Tools and Home Improvement, Toys and Games, Sports and Outdoors"),
        html.Li("Success label: top 10% products within each category by lifetime review count")
    ]),

    html.H2("Analytical Methods"),
    html.Ul([
        html.Li("Data sampling from large Amazon review datasets"),
        html.Li("Feature engineering from the first 4 weeks of reviews"),
        html.Li("Baseline model: Logistic Regression"),
        html.Li("Advanced model: XGBoost"),
        html.Li("Evaluation metrics: ROC-AUC, Average Precision, Recall, F1, and Precision@K")
    ]),

    html.H2("Major Finding 1: Early Data Can Predict Product Success"),
    html.P("""
    Both models performed better than random. XGBoost achieved ROC-AUC around 0.679,
    compared to 0.638 for Logistic Regression. This shows that early review behavior
    contains useful predictive signals.
    """),
    fig("roc_curve.png", "Figure 1: ROC curve comparison between Logistic Regression and XGBoost."),
    fig("pr_curve.png", "Figure 2: Precision-recall comparison. XGBoost performs better on imbalanced data."),
    fig("metric_comparison.png", "Figure 3: Metric comparison across models."),

    html.H2("Major Finding 2: Engagement Features Matter Most"),
    html.P("""
    The strongest predictors are engagement-related features, including total votes,
    votes per review, review count, and review velocity. Rating-based features were less
    informative because many products already have high ratings.
    """),
    fig("correlation_features.png", "Figure 4: Correlation of first-4-week features with product success."),
    fig("xgboost_importance.png", "Figure 5: XGBoost feature importance."),

    html.H2("Major Finding 3: Performance Varies by Category"),
    html.P("""
    Model performance is not uniform across categories. XGBoost performed best for
    Tools and Home Improvement and weaker for Clothing, Shoes, and Jewelry. This suggests
    that future models may need category-specific tuning.
    """),
    fig("category_auc.png", "Figure 6: Category-level ROC-AUC comparison."),
    fig("category_consistency.png", "Figure 7: Model performance consistency across categories."),

    html.H2("System Design"),
    html.P("""
    The website is designed to be hosted on Google App Engine. The dataset and processed
    outputs can be stored in Google Cloud Storage. A trained model can later be deployed
    as a Dockerized inference service on Cloud Run and accessed by the website through an API.
    """),

    html.H2("Next Steps"),
    html.Ul([
        html.Li("Add NLP features such as sentiment analysis and review topics"),
        html.Li("Test different early windows such as 2, 4, and 8 weeks"),
        html.Li("Improve category-specific modeling"),
        html.Li("Deploy an inference API using Docker and Cloud Run")
    ]),

    html.H2("Conclusion"),
    html.P("""
    Early review data provides useful but incomplete signals. Engagement features are the
    strongest predictors, XGBoost performs better than Logistic Regression, and this problem
    is best treated as a ranking task rather than perfect classification.
    """)

], className="container")

if __name__ == "__main__":
    app.run(debug=True)