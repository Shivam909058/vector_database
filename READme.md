# Vector Database GUI

## Introduction
The Vector Database GUI is a Python application that allows users to store, search, view, and manage text entries with their embeddings using a graphical user interface.

## Features
- Add new entries with a key and text.
- Search for entries based on text similarity.
- View all stored entries.
- Delete entries by key.
- Clear the entire database.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/vector-database-gui.git
    cd vector-database-gui
    ```

2. Install the required dependencies:
    ```sh
    pip install transformers torch numpy scikit-learn tk
    ```

3. Run the application:
    ```sh
    python app.py
    ```

## Usage

### Adding an Entry
1. Click the "Add Entry" button.
2. Enter the key and text.

### Searching for an Entry
1. Click the "Search Database" button.
2. Enter the search query.
3. The best match key, associated text, and similarity score will be displayed.

### Viewing All Entries
1. Click the "View All Entries" button.

### Deleting an Entry
1. Click the "Delete Entry" button.
2. Enter the key to delete.

### Clearing the Database
1. Click the "Clear Database" button.

# Vector Database

A simple vector database using transformers and cosine similarity for efficient high-dimensional data handling and search.

## Features

- Store and manage text embeddings.
- Perform similarity search using cosine similarity.
- View, delete, and clear database entries.

## Installation

```sh
pip install vector-database


