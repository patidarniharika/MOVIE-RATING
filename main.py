# %% [markdown]
# # Movie Rating Prediction
# ### Mini Project — B.Tech CSE (Session 2026-27)
# 
# **Goal:** Predict whether a movie is a **"Hit"** (IMDb score ≥ 7) or **"Not a Hit"**, using basic movie details —
# budget, genre, runtime, and number of votes.
# 
# **Dataset:** IMDb 5000 Movie Dataset (`movie_metadata.csv`) — 5,043 real movies with budget, genre, cast, and rating info.
# Place `movie_metadata.csv` in the same folder as this notebook before running.
# 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (7, 5)


# %% [markdown]
# ## 1. Load the Data

# %%
df = pd.read_csv("movie_metadata.csv")
print("Shape:", df.shape)
df.head()


# %% [markdown]
# ## 2. Keep Only the Columns We Need
# To keep things simple, we use just a handful of basic features instead of all 28 columns.

# %%
cols = ["budget", "duration", "genres", "num_voted_users", "title_year", "imdb_score"]
df = df[cols]
df.head()


# %% [markdown]
# ## 3. Basic Cleaning

# %%
print("Missing values before cleaning:")
print(df.isnull().sum())

# Drop rows with missing values 
df = df.dropna()
print("\nShape after dropping missing rows:", df.shape)


# %%
# Genres column has multiple genres separated by '|' e.g. "Action|Adventure|Fantasy"
# For simplicity, keep only the FIRST (primary) genre
df["main_genre"] = df["genres"].apply(lambda x: x.split("|")[0])
df[["genres", "main_genre"]].head()


# %% [markdown]
# ## 4. Create the Target Variable: Hit or Not

# %%
df["hit"] = (df["imdb_score"] >= 7).astype(int)

print(df["hit"].value_counts())
sns.countplot(x="hit", data=df, palette="Set2")
plt.title("Number of Hit vs Not-Hit Movies")
plt.xlabel("Hit (1 = Yes, IMDb score >= 7)")
plt.show()


# %% [markdown]
# ## 5. Exploratory Data Analysis (EDA)

# %%
plt.figure(figsize=(8,5))
sns.histplot(df["imdb_score"], bins=20, kde=True, color="steelblue")
plt.title("Distribution of IMDb Scores")
plt.xlabel("IMDb Score")
plt.show()


# %%
# Top 10 most common genres and their average IMDb score
top_genres = df["main_genre"].value_counts().head(10).index
genre_avg = df[df["main_genre"].isin(top_genres)].groupby("main_genre")["imdb_score"].mean().sort_values()

plt.figure(figsize=(8,6))
genre_avg.plot(kind="barh", color="teal")
plt.title("Average IMDb Score by Genre (Top 10 Genres)")
plt.xlabel("Average IMDb Score")
plt.show()


# %%
plt.figure(figsize=(8,5))
sns.scatterplot(x="budget", y="imdb_score", data=df, alpha=0.4, color="indianred")
plt.title("Budget vs IMDb Score")
plt.xlabel("Budget")
plt.ylabel("IMDb Score")
plt.xscale("log")
plt.show()


# %%
plt.figure(figsize=(8,5))
sns.scatterplot(x="num_voted_users", y="imdb_score", data=df, alpha=0.4, color="darkorange")
plt.title("Number of Votes vs IMDb Score")
plt.xlabel("Number of Voted Users")
plt.ylabel("IMDb Score")
plt.xscale("log")
plt.show()


# %% [markdown]
# **EDA takeaway:**
# Movies with a very high number of votes tend to skew toward higher IMDb scores, and certain genres
# (e.g. Documentary, Biography) tend to average higher ratings than others (e.g. Horror).
# 

# %% [markdown]
# ## 6. Preprocessing for the Model

# %%
model_df = df.drop(columns=["genres", "imdb_score"])  # drop original genres text col and raw score (used to build target)

le = LabelEncoder()
model_df["main_genre"] = le.fit_transform(model_df["main_genre"])

X = model_df.drop(columns=["hit"])
y = model_df["hit"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train shape:", X_train.shape, " Test shape:", X_test.shape)


# %% [markdown]
# ## 7. Train the Model
# A single Decision Tree classifier — simple and easy to explain in a viva.

# %%
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# %%
print(classification_report(y_test, y_pred, target_names=["Not Hit", "Hit"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Hit", "Hit"], yticklabels=["Not Hit", "Hit"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()


# %% [markdown]
# ## 8. Which Features Matter Most?

# %%
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(7,4))
importances.plot(kind="barh", color="seagreen")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()


# %% [markdown]
# ## 9. Summary
# 
# - Used a single, clean dataset (`movie_metadata.csv`) with just 5 basic input features.
# - Defined a simple target: Hit (IMDb score ≥ 7) vs Not Hit.
# - Trained one Decision Tree model and evaluated it with accuracy, a classification report, and a confusion matrix.
# - Checked which features matter most for the prediction.
# 
# 


