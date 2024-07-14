import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Check if a GPU is available and use it if it is
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model and tokenizer from transformers
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)

# Vector database using a dictionary (hashmap)
vector_database = {}


def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()


def add_to_database(key, text):
    embedding = get_embedding(text)[0]
    vector_database[key] = (text, embedding)


def search_database(query):
    query_embedding = get_embedding(query)[0].reshape(1, -1)
    keys = list(vector_database.keys())
    embeddings = np.array([value[1] for value in vector_database.values()])

    similarities = cosine_similarity(query_embedding, embeddings)
    best_match_idx = np.argmax(similarities)
    best_match_key = keys[best_match_idx]

    return best_match_key, similarities[0][best_match_idx]


def view_all_entries():
    if not vector_database:
        messagebox.showinfo("Info", "Database is empty!")
        return

    entries = "\n".join(f"{key}: {value[0]}" for key, value in vector_database.items())
    display_text(entries)


def delete_entry():
    key = simpledialog.askstring("Input", "Enter key to delete:")
    if key in vector_database:
        del vector_database[key]
        messagebox.showinfo("Info", "Entry deleted successfully!")
    else:
        messagebox.showinfo("Info", "Key not found!")


def clear_database():
    vector_database.clear()
    messagebox.showinfo("Info", "Database cleared!")


def display_text(text):
    display_window = tk.Toplevel(root)
    display_window.title("Display")
    text_area = scrolledtext.ScrolledText(display_window, wrap=tk.WORD)
    text_area.insert(tk.INSERT, text)
    text_area.config(state=tk.DISABLED)
    text_area.pack(expand=True, fill='both')


def add_entry():
    key = simpledialog.askstring("Input", "Enter key:")
    text = simpledialog.askstring("Input", "Enter text:")
    if key and text:
        add_to_database(key, text)
        messagebox.showinfo("Info", "Entry added successfully!")


def search_entry():
    query = simpledialog.askstring("Input", "Enter search query:")
    if query:
        best_match_key, similarity = search_database(query)
        best_match_text = vector_database[best_match_key][0]
        messagebox.showinfo("Result",
                            f"Best match key: {best_match_key}\nText: {best_match_text}\nSimilarity: {similarity}")


# GUI Setup
root = tk.Tk()
root.title("Vector Database GUI")
root.geometry("500x500")

add_button = tk.Button(root, text="Add Entry", command=add_entry)
add_button.pack(pady=5)

search_button = tk.Button(root, text="Search Database", command=search_entry)
search_button.pack(pady=5)

view_button = tk.Button(root, text="View All Entries", command=view_all_entries)
view_button.pack(pady=5)

delete_button = tk.Button(root, text="Delete Entry", command=delete_entry)
delete_button.pack(pady=5)

clear_button = tk.Button(root, text="Clear Database", command=clear_database)
clear_button.pack(pady=5)

root.mainloop()
