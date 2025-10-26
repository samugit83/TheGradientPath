# 🧪 30 Complex Test Questions for Text-to-SQL System

This document provides 30 realistic user questions that test all system capabilities: **fuzzy matching** (typo tolerance), **semantic search** (embeddings), **traditional filters**, **aggregations**, and **complex multi-table queries**.

## Legend
- 🔍 **Tests Fuzzy Matching** - Levenshtein distance for typos
- 🧠 **Tests Semantic Search** - Vector embeddings for concepts
- 🔗 **Tests Multi-Join** - Combines multiple tables
- 📊 **Tests Aggregation** - COUNT, AVG, SUM, etc.
- 🎯 **Tests Complex Filters** - Date ranges, price ranges, ratings

---

## Basic Fuzzy Matching (Typo Tolerance)

### 1. 🔍 Author Name with Typo
**Question:** "Find all books by George Orrwell"

**What it tests:** Fuzzy matching on `author.last_name` with 1 typo  
**Expected:** Should find "George Orwell" using Levenshtein distance

---

### 2. 🔍 Book Title with Multiple Typos
**Question:** "Show me the book titled 'Ninteen Eighty For'"

**What it tests:** Fuzzy matching on `books.title` with multiple typos  
**Expected:** Should find "1984" or "Nineteen Eighty-Four"

---

### 3. 🔍 Publisher Name Typo
**Question:** "What books are published by Penguen Random House?"

**What it tests:** Fuzzy matching on `publishers.publisher_name`  
**Expected:** Should find "Penguin Random House"

---

### 4. 🔍 Category Name with Typo
**Question:** "Find books in the Sciance Fiction category"

**What it tests:** Fuzzy matching on `categories.category_name`  
**Expected:** Should find "Science Fiction" category

---

### 5. 🔍 Reviewer Name Search
**Question:** "Show me reviews written by Sara Jonson"

**What it tests:** Fuzzy matching on `reviews.reviewer_name` with typo  
**Expected:** Should find "Sarah Johnson"

---

## Semantic Search (Embeddings)

### 6. 🧠 Book Content Similarity
**Question:** "Find books similar to '1984' about dystopia and government control"

**What it tests:** Semantic search on `books.book_description_embed`  
**Expected:** Uses embedding for "dystopian totalitarian surveillance oppression" concepts

---

### 7. 🧠 Conceptual Book Search
**Question:** "Find books about social issues"

**What it tests:** Semantic search for concepts not exact text  
**Expected:** Searches `book_description_embed` for AI/ML themes

---

### 8. 🧠 Publisher Focus Search
**Question:** "Which publishers focus on children's literature?"

**What it tests:** Semantic search on `publishers.publishing_focus_description_embed`  
**Expected:** Finds publishers semantically matching children's books focus

---

### 9. 🧠 Review Content Search
**Question:** "Find reviews that discuss character development and plot twists"

**What it tests:** Semantic search on `reviews.review_text_embed`  
**Expected:** Finds reviews discussing these narrative elements

---

### 10. 🧠 Thematic Book Search
**Question:** "Show me books with themes similar to The Alchemist about self-discovery"

**What it tests:** Semantic search for abstract themes  
**Expected:** Searches for spiritual journey, personal growth themes

---

## Combined Fuzzy + Semantic

### 11. 🔍🧠 Typo + Semantic Content
**Question:** "Find books by Margret Atwood about dystopian societies"

**What it tests:** Fuzzy match on author name + semantic search on book content  
**Expected:** Levenshtein on "Margaret Atwood" + embedding search for dystopian themes

---

### 12. 🔍🧠 Publisher Typo + Focus
**Question:** "Show books from Tor Boooks that are about space exploration"

**What it tests:** Fuzzy publisher name + semantic publishing focus  
**Expected:** Levenshtein on "Tor Books" + semantic match on sci-fi/space themes

---

### 13. 🔍🧠🔗 Multi-table Fuzzy + Semantic
**Question:** "Find fantasy books by Terry Pratchet published by Bloomsbery"

**What it tests:** Multiple fuzzy matches + category matching  
**Expected:** Fuzzy on author AND publisher + category filter

---

## Aggregations & Statistics

### 14. 📊 Simple Count
**Question:** "How many books are in the database?"

**What it tests:** Basic COUNT aggregation  
**Expected:** `SELECT COUNT(*) FROM books`

---

### 15. 📊 Count by Author with Typo
**Question:** "How many books did Stephan King write?"

**What it tests:** Fuzzy match + COUNT GROUP BY  
**Expected:** Levenshtein on "Stephen King" + COUNT with GROUP BY

---

### 16. 📊 Average Price by Category
**Question:** "What is the average price of Science Fiction books?"

**What it tests:** AVG aggregation with category filter  
**Expected:** JOIN categories + AVG(retail_price) + category name fuzzy match

---

### 17. 📊 Top Rated Books
**Question:** "Show me the top 10 highest rated books based on reviews"

**What it tests:** AVG rating + ORDER BY + LIMIT  
**Expected:** JOIN reviews + AVG(rating) GROUP BY book + ORDER DESC

---

### 18. 📊 Publisher Statistics
**Question:** "Which publishers have released more than 5 books?"

**What it tests:** COUNT + GROUP BY + HAVING  
**Expected:** GROUP BY publisher + HAVING COUNT(*) > 5

---

## Complex Filters

### 19. 🎯 Price Range Filter
**Question:** "Find books priced between $10 and $20 published after 2000"

**What it tests:** Multiple numeric/date filters  
**Expected:** WHERE retail_price BETWEEN 10 AND 20 AND publication_date > '2000-01-01'

---

### 20. 🎯 Date Range + Rating
**Question:** "Show books published in the last 20 years with reviews rated 4 or higher"

**What it tests:** Date arithmetic + JOIN + rating filter  
**Expected:** publication_date >= CURRENT_DATE - INTERVAL '20 years' + JOIN reviews + rating >= 4

---

### 21. 🎯 Multi-Criteria Filter
**Question:** "Find active bestseller books under $15 from American authors"

**What it tests:** Boolean flags + price + nationality filter  
**Expected:** WHERE is_active = TRUE AND is_bestseller = TRUE AND retail_price < 15 + JOIN authors WHERE nationality LIKE '%American%'

---

## Advanced Multi-Table Queries

### 22. 🔗🧠 Books + Reviews + Semantic
**Question:** "Find highly rated books similar to 'Harry Potter' with reviews mentioning magic"

**What it tests:** Multi-JOIN + semantic on books + semantic on reviews  
**Expected:** Dual semantic search on book_description_embed AND review_text_embed + rating filter

---

### 23. 🔗📊 Author + Publisher + Stats
**Question:** "Which author has the most books published by Penguin Random House?"

**What it tests:** Multi-JOIN + GROUP BY + fuzzy publisher name  
**Expected:** JOIN books + publishers + COUNT GROUP BY author + fuzzy match publisher

---

### 24. 🔗🎯 Category + Price + Date
**Question:** "Show me Fantasy books under $25 published by UK publishers after 2010"

**What it tests:** Category JOIN + price + country + date filters  
**Expected:** Multiple JOINs with category, publisher country filter, price and date WHERE clauses

---

### 25. 🔗🔍 Fuzzy Multi-Entity
**Question:** "Find books by Haruki Murakam published by Vintage Boooks"

**What it tests:** Multiple fuzzy matches across tables  
**Expected:** Levenshtein on both author name AND publisher name

---

## Ultra-Complex Combinations

### 26. 🔍🧠🔗📊 Everything Combined
**Question:** "Find books about adventure similar to Treasure Island by authors with names like 'Stevenson', published after 1990, with average rating above 4, grouped by publisher"

**What it tests:** Semantic content + fuzzy author + date + aggregation + multi-JOIN  
**Expected:** 
- Semantic search on book_description_embed for "adventure pirates sailing treasure"
- Fuzzy match on author name
- Date filter publication_date > '1990-01-01'
- JOIN reviews + AVG(rating) >= 4
- GROUP BY publisher

---

### 27. 🧠🔗🎯 Semantic + Complex Filters
**Question:** "Show me dystopian books similar to The Handmaid's Tale, priced under $20, published by feminist-focused publishers founded after 1980"

**What it tests:** Semantic book search + semantic publisher focus + price + date  
**Expected:**
- Semantic on book_description_embed for dystopian themes
- Semantic on publishing_focus_description_embed for feminist literature
- Price filter retail_price < 20
- Publisher year_founded > 1980

---

### 28. 🔍🧠📊 Fuzzy + Semantic + Stats
**Question:** "How many books about space exploration are there by authors with names similar to 'Asimof' or 'Clark'?"

**What it tests:** Semantic content + multiple fuzzy OR conditions + COUNT  
**Expected:**
- Semantic search on book_description_embed for space/exploration themes
- (Levenshtein for "Asimov" OR Levenshtein for "Clarke")
- COUNT aggregation

---

### 29. 🔗🧠🎯📊 Multi-Semantic Search
**Question:** "Find books and reviews where both the book description and review text discuss artificial intelligence, published after 2015, with ratings 4+, showing average rating per book"

**What it tests:** Dual semantic search + date + rating + aggregation  
**Expected:**
- Semantic on book_description_embed for AI concepts
- Semantic on review_text_embed for AI concepts
- Date filter publication_date > '2015-01-01'
- rating >= 4
- AVG(rating) GROUP BY book

---

### 30. 🔍🧠🔗🎯📊 Ultimate Challenge
**Question:** "Compare books similar to both '1984' and 'Brave New World' written by authors with names ending in 'well' or 'ley', published by publishers focusing on literary fiction, priced between $12-$18, with reviews mentioning social commentary, showing average rating and total sales for each book published in the last 30 years"

**What it tests:** Multiple semantic searches + fuzzy patterns + semantic publisher + price range + semantic review + aggregations + date filter + multi-JOIN  
**Expected:**
- Two semantic searches on book_description_embed (1984 themes + Brave New World themes)
- Fuzzy pattern match on author names (LIKE '%well' OR LIKE '%ley')
- Semantic on publishing_focus_description_embed for literary fiction
- Price BETWEEN 12 AND 18
- Semantic on review_text_embed for social commentary
- AVG(rating), SUM(total_sales) GROUP BY book
- publication_date >= CURRENT_DATE - INTERVAL '30 years'
- Multiple JOINs (authors, publishers, reviews)

---

## Testing Guidelines

Each question is designed to test specific system capabilities:

1. **Questions 1-5**: Pure fuzzy matching with Levenshtein distance
2. **Questions 6-10**: Pure semantic search with vector embeddings
3. **Questions 11-13**: Combination of fuzzy matching and semantic search
4. **Questions 14-18**: Aggregations and statistics
5. **Questions 19-21**: Complex filtering conditions
6. **Questions 22-25**: Advanced multi-table JOINs
7. **Questions 26-30**: Ultimate combinations testing all features together

## Expected System Behavior

### ✅ Fuzzy Matching
- System should use `levenshtein(LOWER(field), LOWER('term')) <= threshold`
- Appropriate thresholds: 2 for short strings, 3 for medium, 5 for long
- Results ordered by distance (closest first)

### ✅ Semantic Search
- System should use `field_embed <-> %s::vector` for similarity
- Proper embedding generation for user's search terms
- Check for `IS NOT NULL` on embedding fields
- Order by similarity score (lower distance = more similar)

### ✅ Query Limits
- Semantic searches: LIMIT 10-20
- List queries: LIMIT 50
- Aggregations: appropriate grouping
- Maximum: LIMIT 100 (hard cap unless user specifies more)

### ✅ Proper JOINs
- Correct foreign key relationships
- Appropriate JOIN types (INNER, LEFT as needed)
- No Cartesian products

### ✅ Combined Strategies
- When both fuzzy and semantic are needed, combine with OR
- Order by relevance (distance/similarity scores)
- Apply all filters correctly (AND logic for multiple criteria)

---

## Running Tests

To test the system with these questions:

```bash
# Interactive mode
python main.py --interactive

# Then paste each question from this file
```

Or create an automated test script:

```python
from text_to_sql_agent import AgentTextToSql

agent = AgentTextToSql()
questions = [
    "Find all books by George Orrwell",
    "Show me books similar to 1984 about dystopia",
    # ... add all 30 questions
]

for i, question in enumerate(questions, 1):
    print(f"\n{'='*80}")
    print(f"Question {i}: {question}")
    print('='*80)
    result = agent.process_request_with_execution(question)
    if result['success']:
        print(f"SQL: {result['sql_query']}")
        print(f"Results: {result['query_results']['row_count']} rows")
        print(f"Answer: {result['final_answer']}")
    else:
        print(f"Error: {result['error']}")
```

---

## Success Criteria

A well-functioning Text-to-SQL system should:

1. ✅ Handle all 30 questions without errors
2. ✅ Generate syntactically correct SQL
3. ✅ Use appropriate fuzzy matching for typos
4. ✅ Use semantic search for conceptual queries
5. ✅ Combine strategies when needed
6. ✅ Apply all filters correctly
7. ✅ Return reasonable result sets (proper LIMIT)
8. ✅ Generate accurate natural language answers

Good luck testing! 🚀

