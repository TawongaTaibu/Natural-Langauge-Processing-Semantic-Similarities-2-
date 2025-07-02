import spacy

# Load the medium-sized spaCy model with word vectors.
nlp = spacy.load('en_core_web_md')


def load_movie_descriptions(movies):
    """
    Reads the movie descriptions from a file and returns them as a list.

    :param movies: Path to the file containing movie descriptions (one per line).
    :type movies: str
    :return: List of movie descriptions.
    :rtype: list

    Example:

    >>> movies = load_movie_descriptions('movies.txt')
    >>> print(len(movies))
    100  # Assuming the file has 100 movie descriptions.
    """
    with open('Tawonga Taibu Task 2/movies.txt', 'r') as file:
        movies = file.readlines()

    return movies


def recommend_movie(user_movie_desc, movies):
    """
    Recommends the most similar movie based on the provided movie description.

    This function compares the user's movie description to each movie in the
    dataset and returns the most similar movie based on word vector similarity.

    :param user_movie_desc: Description of the user's movie (e.g., Planet Hulk).
    :type user_movie_desc: str
    :param movies: List of movie descriptions to compare with.
    :type movies: list
    :return: The most similar movie description from the dataset.
    :rtype: str

    Example:

    >>> movies = load_movie_descriptions('movies.txt')
    >>> user_movie_desc = "A sci-fi adventure with gladiators and alien battles."
    >>> recommend_movie(user_movie_desc, movies)
    'Thor: Ragnarok'
    """
    # Convert the user's movie description to a spaCy doc.
    user_movie_doc = nlp(user_movie_desc)

    # Initialize variables to keep track of the most similar movie.
    most_similar_movie = None
    highest_similarity = 0

    # Compare the user's movie description with each movie in the dataset.
    for movie in movies:
        movie_doc = nlp(movie)
        similarity = user_movie_doc.similarity(movie_doc)

        # Check if the current movie has a higher similarity than the previous ones.
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_movie = movie

    return most_similar_movie


def main():
    """
    Main function that runs the movie recommendation system.

    This function loads the movie descriptions, defines the description of
    'Planet Hulk', and then calls the `recommend_movie()` function to find
    the most similar movie based on the user's input.

    :return: Prints the recommended movie based on the similarity with Planet Hulk.
    :rtype: None

    Example:

    >>> main()
    If you liked 'Planet Hulk', you might also enjoy: 'Thor: Ragnarok'
    """
    # Load the movie descriptions.
    movies = load_movie_descriptions('Tawonga Taibu Task 2/movies.txt')

    # Description of Planet Hulk.
    planet_hulk_desc = (
        "Will he save their world or destroy it? When the Hulk becomes too "
        "dangerous for the Earth, the Illuminati trick Hulk into a shuttle "
        "and launch him into space to a planet where the Hulk can live in "
        "peace. Unfortunately, Hulk lands on the planet Sakaar where he is "
        "sold into slavery and trained as a gladiator."
    )

    # Get the most similar movie recommendation.
    recommended_movie = recommend_movie(planet_hulk_desc, movies)

    # Output the result.
    print(
        "If you liked 'Planet Hulk', you might also enjoy: ", recommended_movie
    )


if __name__ == '__main__':
    main()


"""
References:

1. spaCy Documentation - https://spacy.io/usage/vectors-similarity
2. W3Schools Python File Handling - https://www.w3schools.com/python/python_file_handling.asp
3. GeeksforGeeks spaCy Tutorial - https://www.geeksforgeeks.org/nlp-tasks-with-spacy/
"""
