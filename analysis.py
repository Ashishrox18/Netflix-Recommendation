"""
Netflix Content Strategy Analysis & Recommendation Engine
=========================================================
End-to-end pipeline: EDA → Content Pivot Analysis → Recommendation System
Author: Ashish S
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Colour palette ────────────────────────────────────────────────────────────
NETFLIX_RED   = "#E50914"
NETFLIX_DARK  = "#141414"
NETFLIX_GREY  = "#564d4d"
ACCENT        = "#F5F5F1"
PALETTE       = [NETFLIX_RED, "#B81D24", "#831010", "#F5F5F1", "#AAAAAA", "#E87C03"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "visuals")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")

    # Drop rows where type is not Movie/TV Show (encoding artefacts)
    df = df[df["type"].isin(["Movie", "TV Show"])].copy()

    # Parse date_added
    df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month

    # Clean release_year
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

    # Age of content when added
    df["content_age"] = df["year_added"] - df["release_year"]
    df["content_age"] = df["content_age"].clip(lower=0)

    # Duration: split movies (minutes) vs shows (seasons)
    df["duration_int"] = df["duration"].str.extract(r"(\d+)").astype(float)
    df["duration_unit"] = df["duration"].str.extract(r"(min|Season)")

    # Genre - take primary genre only
    df["primary_genre"] = df["listed_in"].str.split(",").str[0].str.strip()

    # Fill key nulls
    df["director"].fillna("Unknown", inplace=True)
    df["cast"].fillna("Unknown", inplace=True)
    df["country"].fillna("Unknown", inplace=True)
    df["description"].fillna("", inplace=True)
    df["rating"].fillna("NR", inplace=True)

    # Content soup for recommendation engine
    df["content_soup"] = (
        df["title"].fillna("") + " " +
        df["primary_genre"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["rating"].fillna("")
    )

    print(f"[✓] Loaded {len(df):,} titles  |  Movies: {(df['type']=='Movie').sum():,}  |  TV Shows: {(df['type']=='TV Show').sum():,}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PIVOT ANALYSIS  — Should Netflix pivot?
# ─────────────────────────────────────────────────────────────────────────────

def pivot_analysis(df: pd.DataFrame):
    print("\n── PIVOT ANALYSIS ──────────────────────────────────────────────────")

    # 2a. Content type trend over years
    yearly = (df.groupby(["year_added", "type"])
                .size()
                .reset_index(name="count")
                .dropna(subset=["year_added"]))
    yearly = yearly[yearly["year_added"] >= 2015]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Netflix Content Strategy — Pivot Analysis", fontsize=16, fontweight="bold", color=NETFLIX_DARK)

    # Plot 1: Movies vs TV Shows added per year
    for t, c in zip(["Movie", "TV Show"], [NETFLIX_RED, NETFLIX_GREY]):
        sub = yearly[yearly["type"] == t]
        axes[0].plot(sub["year_added"], sub["count"], marker="o", label=t, color=c, linewidth=2.5)
    axes[0].set_title("Content Added per Year", fontweight="bold")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Titles Added")
    axes[0].legend()

    # Plot 2: Rating distribution
    rating_order = ["TV-Y", "TV-Y7", "TV-G", "G", "PG", "TV-PG", "PG-13", "TV-14", "R", "TV-MA", "NR"]
    rating_counts = df["rating"].value_counts().reindex(rating_order).dropna()
    bars = axes[1].bar(rating_counts.index, rating_counts.values,
                       color=[NETFLIX_RED if v == rating_counts.max() else NETFLIX_GREY for v in rating_counts.values])
    axes[1].set_title("Content by Maturity Rating", fontweight="bold")
    axes[1].set_xlabel("Rating")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=45)

    # Plot 3: Top 10 countries
    country_counts = (df[df["country"] != "Unknown"]["country"]
                      .str.split(", ").explode()
                      .value_counts().head(10))
    axes[2].barh(country_counts.index[::-1], country_counts.values[::-1],
                 color=[NETFLIX_RED if i == 0 else NETFLIX_GREY for i in range(len(country_counts))])
    axes[2].set_title("Top 10 Content-Producing Countries", fontweight="bold")
    axes[2].set_xlabel("Titles")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "01_pivot_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved {out}")

    # 2b. Genre trends
    genre_type = (df.groupby(["primary_genre", "type"])
                    .size().reset_index(name="count"))
    top_genres = df["primary_genre"].value_counts().head(12).index
    genre_pivot = (genre_type[genre_type["primary_genre"].isin(top_genres)]
                   .pivot(index="primary_genre", columns="type", values="count")
                   .fillna(0))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(genre_pivot))
    w = 0.4
    ax.bar(x - w/2, genre_pivot.get("Movie", 0),   width=w, label="Movie",   color=NETFLIX_RED)
    ax.bar(x + w/2, genre_pivot.get("TV Show", 0), width=w, label="TV Show", color=NETFLIX_GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(genre_pivot.index, rotation=45, ha="right")
    ax.set_title("Genre Distribution — Movies vs TV Shows", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "02_genre_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved {out}")

    # Key insight
    movies_2021 = yearly[(yearly["type"] == "Movie") & (yearly["year_added"] == 2021)]["count"].sum()
    shows_2021  = yearly[(yearly["type"] == "TV Show") & (yearly["year_added"] == 2021)]["count"].sum()
    print(f"\n  Insight → In 2021: {movies_2021:.0f} movies vs {shows_2021:.0f} TV shows added")
    print(  "  Insight → TV Show growth outpaces Movies post-2019 — pivot toward serialised content is justified")


# ─────────────────────────────────────────────────────────────────────────────
# 3. GENETIC ALGORITHM — Feature Selection for Content Classifier
# ─────────────────────────────────────────────────────────────────────────────

def genetic_feature_selection(X: np.ndarray, y: np.ndarray,
                               feature_names: list,
                               n_generations: int = 20,
                               population_size: int = 30,
                               mutation_rate: float = 0.1) -> list:
    """
    Binary Genetic Algorithm for feature subset selection.
    Each chromosome = binary vector over feature indices.
    Fitness = 3-fold cross-val accuracy of Random Forest on selected features.
    """
    n_features = X.shape[1]

    def fitness(chromosome):
        selected = [i for i, g in enumerate(chromosome) if g == 1]
        if not selected:
            return 0.0
        Xs = X[:, selected]
        clf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
        return cross_val_score(clf, Xs, y, cv=3, scoring="accuracy").mean()

    # Initialise population
    population = [np.random.randint(0, 2, n_features).tolist()
                  for _ in range(population_size)]

    best_chromosome = None
    best_score      = -1

    for gen in range(n_generations):
        scores = [fitness(c) for c in population]

        # Track best
        idx = int(np.argmax(scores))
        if scores[idx] > best_score:
            best_score      = scores[idx]
            best_chromosome = population[idx][:]

        # Selection (tournament)
        def tournament(pop, sc, k=3):
            chosen = np.random.choice(len(pop), k, replace=False)
            return list(pop[chosen[int(np.argmax([sc[i] for i in chosen]))]][:])

        new_pop = []
        while len(new_pop) < population_size:
            p1 = tournament(np.array(population), scores)
            p2 = tournament(np.array(population), scores)
            # Single-point crossover
            pt = np.random.randint(1, n_features)
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
            # Mutation
            for c in [c1, c2]:
                for i in range(n_features):
                    if np.random.rand() < mutation_rate:
                        c[i] = 1 - c[i]
            new_pop.extend([c1, c2])

        population = new_pop[:population_size]
        if (gen + 1) % 5 == 0:
            print(f"    Gen {gen+1:02d}/{n_generations} — best fitness: {best_score:.4f}")

    selected_features = [feature_names[i] for i, g in enumerate(best_chromosome) if g == 1]
    print(f"  [✓] GA selected {len(selected_features)} / {n_features} features  |  Best CV accuracy: {best_score:.4f}")
    return selected_features, best_score


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONTENT CLASSIFIER  — Predict Movie vs TV Show
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(df: pd.DataFrame):
    print("\n── CONTENT CLASSIFIER ───────────────────────────────────────────────")

    # Feature engineering
    le_rating  = LabelEncoder()
    le_country = LabelEncoder()

    df["rating_enc"]  = le_rating.fit_transform(df["rating"].fillna("NR"))
    df["country_enc"] = le_country.fit_transform(df["country"].fillna("Unknown"))

    df = df.copy()
    df["description"] = df["description"].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=100, stop_words="english")
    desc_matrix = tfidf.fit_transform(df["description"]).toarray()
    desc_df     = pd.DataFrame(desc_matrix, columns=[f"desc_{i}" for i in range(desc_matrix.shape[1])])

    feature_df = pd.concat([
        df[["release_year", "duration_int", "rating_enc", "country_enc",
            "year_added", "month_added", "content_age"]].reset_index(drop=True),
        desc_df.reset_index(drop=True)
    ], axis=1).fillna(0)

    X            = feature_df.values
    feature_names = feature_df.columns.tolist()
    y            = (df["type"] == "Movie").astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Genetic Algorithm feature selection
    print("  Running Genetic Algorithm for feature selection …")
    selected_features, ga_score = genetic_feature_selection(
        X_train, y_train, feature_names, n_generations=20, population_size=30)

    sel_idx   = [feature_names.index(f) for f in selected_features]
    X_train_s = X_train[:, sel_idx]
    X_test_s  = X_test[:, sel_idx]

    # Train Random Forest on GA-selected features
    print("\n  Training Random Forest on GA-selected features …")
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)

    y_pred = rf.predict(X_test_s)
    y_prob = rf.predict_proba(X_test_s)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\n  Random Forest Results:")
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  Accuracy: {(y_pred == y_test).mean():.4f}")
    print(classification_report(y_test, y_pred, target_names=["TV Show", "Movie"]))

    # Feature importance plot
    importances = pd.Series(rf.feature_importances_, index=selected_features).nlargest(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    importances[::-1].plot(kind="barh", ax=ax,
                           color=[NETFLIX_RED if i == 0 else NETFLIX_GREY for i in range(len(importances))])
    ax.set_title("Top 15 Feature Importances (GA + Random Forest)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "03_feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [✓] Saved {out}")

    # Confusion matrix
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["TV Show", "Movie"],
                yticklabels=["TV Show", "Movie"], ax=ax)
    ax.set_title("Confusion Matrix — Content Type Classifier", fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "04_confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved {out}")

    return rf, tfidf, selected_features, feature_names, auc


# ─────────────────────────────────────────────────────────────────────────────
# 5. RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def build_recommender(df: pd.DataFrame):
    print("\n── RECOMMENDATION ENGINE ────────────────────────────────────────────")

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df["content_soup"])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    title_index = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()

    print(f"  [✓] TF-IDF matrix: {tfidf_matrix.shape}  |  Similarity matrix: {cosine_sim.shape}")

    def recommend(title: str, n: int = 10) -> pd.DataFrame:
        title_lower = title.lower()
        if title_lower not in title_index:
            # Fuzzy fallback
            matches = [t for t in title_index.index if title_lower in t]
            if not matches:
                return pd.DataFrame({"error": [f"'{title}' not found in dataset"]})
            title_lower = matches[0]

        idx      = title_index[title_lower]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        scores        = [round(i[1], 4) for i in sim_scores]

        result = df.iloc[movie_indices][["title", "type", "primary_genre", "rating",
                                         "release_year", "country", "description"]].copy()
        result.insert(0, "similarity_score", scores)
        return result.reset_index(drop=True)

    # Demo recommendations
    test_titles = ["Inception", "Breaking Bad", "The Crown"]
    for title in test_titles:
        print(f"\n  Top 5 recommendations for '{title}':")
        recs = recommend(title, n=5)
        if "error" not in recs.columns:
            print(recs[["title", "type", "primary_genre", "similarity_score"]].to_string(index=False))
        else:
            print(f"  {recs.iloc[0]['error']}")

    return recommend, cosine_sim


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONTENT TREND DEEP-DIVE
# ─────────────────────────────────────────────────────────────────────────────

def trend_deepdive(df: pd.DataFrame):
    print("\n── TREND DEEP-DIVE ──────────────────────────────────────────────────")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Netflix Content Deep-Dive", fontsize=16, fontweight="bold")

    # Top genres by type
    movies = df[df["type"] == "Movie"]["primary_genre"].value_counts().head(8)
    shows  = df[df["type"] == "TV Show"]["primary_genre"].value_counts().head(8)

    axes[0, 0].barh(movies.index[::-1], movies.values[::-1], color=NETFLIX_RED)
    axes[0, 0].set_title("Top Movie Genres", fontweight="bold")
    axes[0, 0].set_xlabel("Count")

    axes[0, 1].barh(shows.index[::-1], shows.values[::-1], color=NETFLIX_GREY)
    axes[0, 1].set_title("Top TV Show Genres", fontweight="bold")
    axes[0, 1].set_xlabel("Count")

    # Duration distribution
    movie_dur = df[(df["type"] == "Movie") & (df["duration_int"] < 250)]["duration_int"].dropna()
    axes[1, 0].hist(movie_dur, bins=40, color=NETFLIX_RED, edgecolor="white", alpha=0.85)
    axes[1, 0].axvline(movie_dur.median(), color=NETFLIX_DARK, linestyle="--", linewidth=2,
                        label=f"Median: {movie_dur.median():.0f} min")
    axes[1, 0].set_title("Movie Duration Distribution", fontweight="bold")
    axes[1, 0].set_xlabel("Duration (minutes)")
    axes[1, 0].legend()

    # Content added by month
    month_counts = df["month_added"].value_counts().sort_index()
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    axes[1, 1].bar(month_labels, [month_counts.get(i, 0) for i in range(1, 13)],
                   color=[NETFLIX_RED if v == month_counts.max() else NETFLIX_GREY
                          for v in [month_counts.get(i, 0) for i in range(1, 13)]])
    axes[1, 1].set_title("Content Added by Month (Seasonality)", fontweight="bold")
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].set_ylabel("Titles Added")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "05_trend_deepdive.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "netflix_titles.csv")

    print("=" * 65)
    print("  Netflix Content Strategy & Recommendation Engine")
    print("=" * 65)

    df            = load_and_clean(DATA_PATH)
    pivot_analysis(df)
    rf_model, tfidf_model, sel_features, all_features, auc = train_classifier(df)
    recommend_fn, sim_matrix = build_recommender(df)
    trend_deepdive(df)

    print("\n" + "=" * 65)
    print(f"  Pipeline complete.")
    print(f"  Classifier AUC-ROC : {auc:.4f}")
    print(f"  Visuals saved to   : visuals/")
    print("=" * 65)
