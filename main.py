
import time
import functools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        ret = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"{func.__name__} completed in {t1-t0} seconds")
        return ret
    return wrapper


@timer
def read(filename: str, **kwargs) -> pd.DataFrame:
    with open(f"data\\{filename}.csv", encoding="utf-8") as file:
        df = pd.read_csv(file, engine="python", **kwargs)
    return df


def print_info(df: pd.DataFrame):
    print(df.keys())
    print(df.shape)
    df.info()
    print()
    print(df.head())


def get_movie_id(title):
    titles = read("movie_titles")
    return titles["movieId"][(titles["title"] == title)].iloc[0]


def get_movie_titles(ids):
    ret = []
    print(ids)
    movie_titles = read("movie_titles")
    movie_titles.set_index("movieId", inplace=True)
    for movieId in ids:
        ret.append(movie_titles.loc[movieId, "title"])

    return ret


def create_genome_pivot():
    df = read("genome-scores")
    df = df.pivot(index="movieId", columns="tagId", values="relevance")

    print_info(df)
    df.to_csv(r"data\genome-scores-pivot.csv")


def genome_scores_pca_explore():
    df = read("genome-scores-pivot")

    pca = PCA()
    scaler = StandardScaler()

    df_scaled = scaler.fit_transform(df)
    pca.fit_transform(df_scaled)
    pca_variance: np.ndarray = pca.explained_variance_ratio_

    plt.figure(figsize=(16, 9))
    plt.bar(range(1129), pca_variance, alpha=0.5, align='center', label='Explained Variance Ratio')
    # plt.legend()
    plt.ylabel("Variance ratio")
    plt.xlabel("Principal Component")
    plt.show()

    for target_variance_explained in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
        print(f"{target_variance_explained}: {sum(pca_variance.cumsum() < target_variance_explained)}")

    plt.cla()
    plt.plot(
        range(len(pca_variance)), pca_variance.cumsum(), alpha=0.5, label="Cumulative Sum of Explained Variance"
    )
    # plt.legend()
    plt.ylabel("Cumulative Sum of Explained Variance")
    plt.xlabel("Number of PCs")
    plt.show()


def ratings_pivot() -> pd.DataFrame:
    """Returns ratings data as a sparse pivot table."""
    df: pd.DataFrame = read("ratings")

    # Discarding timestamp and converting rating variable from float64 to int.
    df.drop("timestamp", axis="columns", inplace=True)
    df["rating"] = df["rating"].mul(2, fill_value=0)
    df = df.astype({"userId": int, "movieId": int, "rating": np.uint8})

    user_counts = df.groupby("userId").agg("count")["rating"]
    movie_counts = df.groupby("movieId").agg("count")["rating"]
    low_user_counts = user_counts.index[(user_counts < 40)]
    low_movie_counts = movie_counts.index[(movie_counts < 120)]

    df = df[(~df["userId"].isin(low_user_counts)) & (~df["movieId"].isin(low_movie_counts))]

    df = df.pivot(index="userId", columns="movieId", values="rating")

    df.fillna(0, inplace=True)

    return df.astype("uint8")


def get_similar_movies_CF(user_ratings, similarity_df: pd.DataFrame):
    # (similarity_df[movieID] * (user_rating - 2.5)).sort_values(ascending=False)

    similar_movies = pd.DataFrame()

    for movieID, user_rating in user_ratings:
        try:
            similar_movies = similar_movies.append(similarity_df[movieID] * (2 * user_rating - 4), ignore_index=True)
        except KeyError:
            continue

    return similar_movies.sum().sort_values(ascending=False)


def transform_movies():
    df = read("movies")
    df.set_index("movieId", inplace=True)

    title: pd.Series = df["title"].str.rpartition("(").iloc[:, 0]
    year: pd.Series = df["title"].str.rpartition("(").iloc[:, 2]

    def convert_to_int(x):
        try:
            return int(x)
        except ValueError:
            return -1

    year = year.str.strip("() ").apply(convert_to_int)

    mean_year = int(year[(year != -1)].mean())
    year.replace(-1, mean_year)

    df["title"] = title.str.strip()
    df["year"] = year

    # genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    #           "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    # for genre in genres:
    #     df[genre] = df["genres"].str.contains(genre).apply(int)
    # df.drop("genres", axis="columns", inplace=True)

    df["title"].to_csv(r"data\movie_titles.csv")

    df['genres'] = df['genres'].str.replace("|", " ")
    df['genres'] = df['genres'].str.replace("Sci-Fi", "SciFi")
    df['genres'] = df['genres'].str.replace("Film-Noir", "FilmNoir")
    df.to_csv(r"data\movies_transformed.csv")


def apply_genome_pca():
    tag_genome: pd.DataFrame = read("genome-scores-pivot")
    tag_genome.set_index("movieId", inplace=True)

    pca = PCA(n_components=238)
    genome_pca = pca.fit_transform(tag_genome)

    genome_pca = pd.DataFrame(genome_pca)
    print_info(genome_pca)

    movie_similarity_CBF = genome_pca.T.corr()

    print("saving")
    movie_similarity_CBF.to_csv(r"data\similarity_CBF.csv")


def similarity_matrix_CB():
    df = read("movies_transformed")
    tfdif_vectorizer = TfidfVectorizer(stop_words="english")
    tfdif_matrix = tfdif_vectorizer.fit_transform(df["genres"])

    similarity_matrix = cosine_similarity(tfdif_matrix, tfdif_matrix)

    return pd.DataFrame(similarity_matrix, index=df["movieId"], columns=df["movieId"])


def get_CBF_recommendations_from_movie(movieId, n=10):
    sim_matrix = similarity_matrix_CB()
    movie_list = list(zip(sim_matrix.index, sim_matrix[movieId]))
    similar_movies = list(filter(lambda x: x[0] != movieId, sorted(movie_list, key=lambda x: x[1], reverse=True)))

    # print(similar_movies)

    recommendations = []
    for i in range(n):
        recommendations.append(similar_movies[i][0])

    return recommendations


def get_CBF_recommendations():
    recommendations = get_CBF_recommendations_from_movie(get_movie_id("Finding Nemo"))

    print(get_movie_titles(recommendations))


def get_CF_recommendations_from_movie(movieId, n=10):
    ratings_df = ratings_pivot()

    print_info(ratings_df)
    sim_matrix = ratings_df.corr()

    plt.imshow(sim_matrix)
    plt.title("Collaborative Filtering")
    plt.xlabel("Movie ID")
    plt.ylabel("Movie ID")
    plt.show()

    movie_list = list(zip(sim_matrix.index, sim_matrix[movieId]))
    similar_movies = list(filter(lambda x: x[0] != movieId, sorted(movie_list, key=lambda x: x[1], reverse=True)))

    # print(similar_movies)

    recommendations = []
    for i in range(n):
        recommendations.append(similar_movies[i][0])

    return recommendations


def get_CF_recommendations():
    recommendations = get_CF_recommendations_from_movie(get_movie_id("Kill Bill: Vol. 1"))

    print(get_movie_titles(recommendations))


def get_tag_genome_recommendations_from_movie(movieId, n=10):
    tag_genome = read("genome-scores-pivot")

    sim_matrix = cosine_similarity(tag_genome, tag_genome)
    sim_matrix = pd.DataFrame(sim_matrix, index=sim_matrix["movieId"], columns=sim_matrix["movieId"])
    movie_list = list(zip(sim_matrix.index, sim_matrix[movieId]))
    similar_movies = list(filter(lambda x: x[0] != movieId, sorted(movie_list, key=lambda x: x[1], reverse=True)))

    print(similar_movies)

    recommendations = []
    for i in range(n):
        recommendations.append(similar_movies[i][0])

    return recommendations


def get_tag_genome_recommendations():
    recommendations = get_tag_genome_recommendations_from_movie(get_movie_id("Toy Story"))

    print(get_movie_titles(recommendations))


if __name__ == '__main__':
   print_info(ratings_pivot())
