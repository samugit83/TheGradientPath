-- ============================================================================
-- BOOKS DATABASE SCHEMA
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "fuzzystrmatch";

-- ============================================================================
-- TABLE 1: AUTHORS
-- ============================================================================
CREATE TABLE authors (
    author_id SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    pen_name VARCHAR(150),
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(20),
    date_of_birth DATE,
    date_of_death DATE,
    nationality VARCHAR(100),
    biography TEXT,
    literary_style_description TEXT,
    awards_received TEXT,
    website_url VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_authors_last_name ON authors(last_name);
CREATE INDEX idx_authors_active ON authors(is_active);

COMMENT ON TABLE authors IS 'Stores comprehensive information about book authors';
COMMENT ON COLUMN authors.author_id IS 'PRIMARY KEY: Auto-incrementing unique identifier';


-- ============================================================================
-- TABLE 2: PUBLISHERS
-- ============================================================================
CREATE TABLE publishers (
    publisher_id SERIAL PRIMARY KEY,
    publisher_name VARCHAR(200) NOT NULL UNIQUE,
    legal_name VARCHAR(250),
    registration_number VARCHAR(50),
    tax_id VARCHAR(50),
    email VARCHAR(255),
    phone VARCHAR(20),
    website_url VARCHAR(500),
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state_province VARCHAR(100),
    postal_code VARCHAR(20),
    country VARCHAR(100),
    company_description TEXT,
    company_description_embed vector(1536),
    publishing_focus_description TEXT,
    publishing_focus_description_embed vector(1536),
    year_founded INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_publishers_name ON publishers(publisher_name);
CREATE INDEX idx_publishers_country ON publishers(country);

COMMENT ON TABLE publishers IS 'Stores comprehensive information about publishing companies';
COMMENT ON COLUMN publishers.publisher_id IS 'PRIMARY KEY: Auto-incrementing unique identifier';


-- ============================================================================
-- TABLE 3: CATEGORIES
-- ============================================================================
CREATE TABLE categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(150) NOT NULL UNIQUE,
    category_code VARCHAR(20) UNIQUE,
    category_description TEXT,
    category_description_embed vector(1536),
    typical_content_description TEXT,
    typical_content_description_embed vector(1536),
    age_group VARCHAR(50),
    reading_level VARCHAR(50),
    popularity_score DECIMAL(5,2) DEFAULT 0.00,
    display_order INTEGER DEFAULT 0,
    icon_url VARCHAR(500),
    banner_image_url VARCHAR(500),
    color_code VARCHAR(7),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_categories_active ON categories(is_active);

COMMENT ON TABLE categories IS 'Stores book categories with hierarchical support';
COMMENT ON COLUMN categories.category_id IS 'PRIMARY KEY: Auto-incrementing unique identifier';


-- ============================================================================
-- TABLE 4: BOOKS
-- ============================================================================
CREATE TABLE books (
    book_id SERIAL PRIMARY KEY,
    isbn_10 VARCHAR(10) UNIQUE,
    isbn_13 VARCHAR(13) UNIQUE,
    title VARCHAR(500) NOT NULL,
    subtitle VARCHAR(500),
    author_id INTEGER NOT NULL,
    publisher_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    publication_date DATE,
    edition VARCHAR(50),
    language VARCHAR(50) DEFAULT 'English',
    page_count INTEGER,
    dimensions VARCHAR(100),
    weight_grams DECIMAL(10,2),
    book_description TEXT,
    book_description_embed vector(1536),
    detailed_summary TEXT,
    detailed_summary_embed vector(1536),
    retail_price DECIMAL(10,2),
    cost_price DECIMAL(10,2),
    discount_percentage DECIMAL(5,2) DEFAULT 0.00,
    table_of_contents TEXT,
    table_of_contents_embed vector(1536),
    total_sales INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    is_bestseller BOOLEAN DEFAULT FALSE,
    CONSTRAINT fk_book_author FOREIGN KEY (author_id) REFERENCES authors(author_id) ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT fk_book_publisher FOREIGN KEY (publisher_id) REFERENCES publishers(publisher_id) ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT fk_book_category FOREIGN KEY (category_id) REFERENCES categories(category_id) ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT chk_positive_price CHECK (retail_price >= 0),
    CONSTRAINT chk_valid_discount CHECK (discount_percentage >= 0 AND discount_percentage <= 100)
);

CREATE INDEX idx_books_author ON books(author_id);
CREATE INDEX idx_books_publisher ON books(publisher_id);
CREATE INDEX idx_books_category ON books(category_id);
CREATE INDEX idx_books_title ON books(title);

COMMENT ON TABLE books IS 'Main table storing comprehensive book information';
COMMENT ON COLUMN books.book_id IS 'PRIMARY KEY: Auto-incrementing unique identifier';
COMMENT ON COLUMN books.author_id IS 'FOREIGN KEY: References authors.author_id';
COMMENT ON COLUMN books.publisher_id IS 'FOREIGN KEY: References publishers.publisher_id';
COMMENT ON COLUMN books.category_id IS 'FOREIGN KEY: References categories.category_id';


-- ============================================================================
-- TABLE 5: REVIEWS
-- ============================================================================
CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    book_id INTEGER NOT NULL,
    reviewer_name VARCHAR(150) NOT NULL,
    reviewer_email VARCHAR(255),
    verified_purchase BOOLEAN DEFAULT FALSE,
    rating INTEGER NOT NULL,
    review_title VARCHAR(255),
    review_text TEXT NOT NULL,
    review_text_embed vector(1536),
    detailed_feedback TEXT,
    detailed_feedback_embed vector(1536),
    review_date DATE DEFAULT CURRENT_DATE,
    helpful_count INTEGER DEFAULT 0,
    not_helpful_count INTEGER DEFAULT 0,
    is_verified BOOLEAN DEFAULT FALSE,
    is_published BOOLEAN DEFAULT TRUE,
    moderation_notes TEXT,
    moderation_notes_embed vector(1536),
    moderator_id INTEGER,
    reading_duration_days INTEGER,
    would_recommend BOOLEAN,
    reader_age_group VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_review_book FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT chk_valid_review_rating CHECK (rating >= 1 AND rating <= 5)
);

CREATE INDEX idx_reviews_book ON reviews(book_id);
CREATE INDEX idx_reviews_rating ON reviews(rating);

COMMENT ON TABLE reviews IS 'Stores customer reviews and ratings for books';
COMMENT ON COLUMN reviews.review_id IS 'PRIMARY KEY: Auto-incrementing unique identifier';
COMMENT ON COLUMN reviews.book_id IS 'FOREIGN KEY: References books.book_id';


-- ============================================================================
-- SAMPLE DATA INSERTION
-- ============================================================================

-- Insert 30+ Authors
INSERT INTO authors (first_name, last_name, pen_name, email, date_of_birth, nationality, biography, literary_style_description, awards_received) VALUES
('George', 'Orwell', 'George Orwell', 'george@orwell.com', '1903-06-25', 'British', 
'Eric Arthur Blair, known by his pen name George Orwell, was an English novelist, essayist, journalist, and critic. His work is characterized by lucid prose, social criticism, opposition to totalitarianism, and support of democratic socialism.',
'Orwell''s writing style is characterized by clear, direct prose free of literary jargon. He advocated for political writing that was transparent and accessible.',
'Prometheus Hall of Fame Award'),

('Jane', 'Austen', NULL, 'jane@austen.com', '1775-12-16', 'British',
'Jane Austen was an English novelist known primarily for her six major novels, which interpret, critique and comment upon the British landed gentry at the end of the 18th century.',
'Austen''s writing is characterized by biting irony, realism, and social commentary delivered through wit and humor.',
'Posthumous recognition as one of the greatest writers in English literature'),

('Gabriel', 'García Márquez', 'Gabo', 'gabriel@garcia.com', '1927-03-06', 'Colombian',
'Gabriel José de la Concordia García Márquez was a Colombian novelist, awarded the 1982 Nobel Prize in Literature.',
'García Márquez is renowned for pioneering magical realism, blending the mundane with the magical.',
'Nobel Prize in Literature (1982)'),

('Stephen', 'King', NULL, 'stephen@king.com', '1947-09-21', 'American',
'Stephen Edwin King is an American author of horror, supernatural fiction, suspense, crime, science-fiction, and fantasy novels.',
'King''s books have been described as relentlessly paced, immersive narratives that explore psychological horror and supernatural themes.',
'Bram Stoker Awards, World Fantasy Award'),

('J.K.', 'Rowling', NULL, 'jk@rowling.com', '1965-07-31', 'British',
'Joanne Rowling, better known by her pen name J.K. Rowling, is a British author best known for writing the Harry Potter fantasy series.',
'Rowling''s writing features rich world-building, complex characters, and intricate plots that appeal to both children and adults.',
'Hugo Award, British Book Awards'),

('Agatha', 'Christie', NULL, 'agatha@christie.com', '1890-09-15', 'British',
'Dame Agatha Mary Clarissa Christie was an English writer known for her 66 detective novels and 14 short story collections.',
'Christie''s writing is known for ingenious plot twists, red herrings, and the puzzle-like structure of her mysteries.',
'Grand Master Award from the Mystery Writers of America'),

('Ernest', 'Hemingway', NULL, 'ernest@hemingway.com', '1899-07-21', 'American',
'Ernest Miller Hemingway was an American novelist, short-story writer, and journalist known for his economical and understated style.',
'Hemingway''s distinctive writing style is characterized by economy and understatement, using simple sentences and minimal adjectives.',
'Nobel Prize in Literature (1954), Pulitzer Prize'),

('Toni', 'Morrison', NULL, 'toni@morrison.com', '1931-02-18', 'American',
'Toni Morrison was an American novelist noted for her examination of Black experience within the Black community.',
'Morrison''s writing is distinguished by its poetic language, innovative narrative structures, and exploration of African American identity.',
'Nobel Prize in Literature (1993), Pulitzer Prize'),

('Haruki', 'Murakami', NULL, 'haruki@murakami.com', '1949-01-12', 'Japanese',
'Haruki Murakami is a Japanese writer whose works have been translated into 50 languages and sold millions of copies internationally.',
'Murakami''s writing blends popular culture, magical realism, and themes of alienation with surreal and dreamlike narratives.',
'Franz Kafka Prize, Jerusalem Prize'),

('Margaret', 'Atwood', NULL, 'margaret@atwood.com', '1939-11-18', 'Canadian',
'Margaret Eleanor Atwood is a Canadian poet, novelist, literary critic, and essayist known for her works of speculative fiction.',
'Atwood''s prose is characterized by sharp wit, social commentary, and exploration of gender, identity, and power dynamics.',
'Booker Prize, Arthur C. Clarke Award'),

('Neil', 'Gaiman', NULL, 'neil@gaiman.com', '1960-11-10', 'British',
'Neil Richard MacKinnon Gaiman is an English author of short fiction, novels, comic books, graphic novels, and films.',
'Gaiman''s writing seamlessly blends mythology, fantasy, and horror with contemporary settings and deeply human characters.',
'Hugo Award, Nebula Award, Newbery Medal'),

('Chimamanda Ngozi', 'Adichie', NULL, 'chimamanda@adichie.com', '1977-09-15', 'Nigerian',
'Chimamanda Ngozi Adichie is a Nigerian writer whose novels address the Biafran war in Nigeria and feminist themes.',
'Adichie''s writing is characterized by elegant prose, nuanced character development, and exploration of identity, race, and gender.',
'MacArthur Fellowship, National Book Critics Circle Award'),

('Kazuo', 'Ishiguro', NULL, 'kazuo@ishiguro.com', '1954-11-08', 'British',
'Sir Kazuo Ishiguro is a British novelist, screenwriter, and short-story writer of Japanese origin.',
'Ishiguro''s writing features unreliable narrators, memory, time, and self-delusion explored through restrained, elegant prose.',
'Nobel Prize in Literature (2017), Booker Prize'),

('Paulo', 'Coelho', NULL, 'paulo@coelho.com', '1947-08-24', 'Brazilian',
'Paulo Coelho de Souza is a Brazilian lyricist and novelist best known for his novel The Alchemist.',
'Coelho''s writing is characterized by spiritual themes, allegory, and simple, accessible prose that conveys philosophical ideas.',
'Crystal Award from the World Economic Forum'),

('Virginia', 'Woolf', NULL, 'virginia@woolf.com', '1882-01-25', 'British',
'Adeline Virginia Woolf was an English writer considered one of the most important modernist 20th-century authors.',
'Woolf''s writing pioneered the use of stream of consciousness as a narrative device and explored themes of consciousness and time.',
'Femina Vie Heureuse Prize'),

('Ray', 'Bradbury', NULL, 'ray@bradbury.com', '1920-08-22', 'American',
'Ray Douglas Bradbury was an American author and screenwriter known for his science fiction and fantasy works.',
'Bradbury''s lyrical writing style combines poetic imagery with speculative fiction, creating evocative and imaginative narratives.',
'National Medal of Arts, Pulitzer Prize Special Citation'),

('Ursula K.', 'Le Guin', NULL, 'ursula@leguin.com', '1929-10-21', 'American',
'Ursula Kroeber Le Guin was an American author best known for her works of speculative fiction and Earthsea fantasy series.',
'Le Guin''s writing explores anthropological, sociological themes through richly imagined worlds and thoughtful, philosophical prose.',
'Hugo Award, Nebula Award, National Book Award'),

('Salman', 'Rushdie', NULL, 'salman@rushdie.com', '1947-06-19', 'British-Indian',
'Sir Ahmed Salman Rushdie is a British Indian novelist and essayist whose work combines magical realism with historical fiction.',
'Rushdie''s writing is characterized by postmodernist techniques, magical realism, and exploration of connections between East and West.',
'Booker Prize, Best of the Booker'),

('Jhumpa', 'Lahiri', NULL, 'jhumpa@lahiri.com', '1967-07-11', 'American',
'Nilanjana Sudeshna "Jhumpa" Lahiri is an American author known for her short stories and novels about Indian Americans.',
'Lahiri''s precise, evocative prose explores themes of identity, displacement, and the immigrant experience with sensitivity and insight.',
'Pulitzer Prize for Fiction'),

('Colson', 'Whitehead', NULL, 'colson@whitehead.com', '1969-11-06', 'American',
'Arch Colson Chipp Whitehead is an American novelist known for his genre-bending literary fiction.',
'Whitehead''s writing combines historical fiction with speculative elements, addressing race and American history with innovation.',
'Pulitzer Prize for Fiction (twice), National Book Award'),

('Octavia', 'Butler', NULL, 'octavia@butler.com', '1947-06-22', 'American',
'Octavia Estelle Butler was an American science fiction author noted for her feminist and Afrofuturist themes.',
'Butler''s writing explores themes of power, race, gender, and humanity through complex, thought-provoking science fiction narratives.',
'Hugo Award, Nebula Award, MacArthur Fellowship'),

('Isaac', 'Asimov', NULL, 'isaac@asimov.com', '1920-01-02', 'American',
'Isaac Asimov was an American writer and professor of biochemistry, prolific in science fiction and popular science books.',
'Asimov''s clear, logical writing style makes complex scientific concepts accessible while crafting imaginative science fiction worlds.',
'Hugo Award, Nebula Grand Master Award'),

('Philip K.', 'Dick', NULL, 'philip@dick.com', '1928-12-16', 'American',
'Philip Kindred Dick was an American science fiction writer known for exploring philosophical and social themes in his work.',
'Dick''s writing is characterized by paranoid narratives, exploration of reality and perception, and prescient technological insights.',
'Hugo Award'),

('Kurt', 'Vonnegut', NULL, 'kurt@vonnegut.com', '1922-11-11', 'American',
'Kurt Vonnegut Jr. was an American writer known for his satirical and darkly humorous novels.',
'Vonnegut''s distinctive style features dark comedy, science fiction elements, and sharp social commentary with accessible prose.',
'Guggenheim Fellowship'),

('Douglas', 'Adams', NULL, 'douglas@adams.com', '1952-03-11', 'British',
'Douglas Noel Adams was an English author, humorist, and satirist best known for The Hitchhiker''s Guide to the Galaxy.',
'Adams'' writing combines absurdist humor, satire, and science fiction with witty, quotable prose.',
'Golden Pan award'),

('Terry', 'Pratchett', NULL, 'terry@pratchett.com', '1948-04-28', 'British',
'Sir Terence David John Pratchett was an English humorist, satirist, and author of fantasy novels, especially the Discworld series.',
'Pratchett''s writing features sharp satire, wordplay, and humanist philosophy wrapped in accessible fantasy narratives.',
'Carnegie Medal, Locus Award'),

('J.R.R.', 'Tolkien', NULL, 'jrr@tolkien.com', '1892-01-03', 'British',
'John Ronald Reuel Tolkien was an English writer, poet, philologist, and academic, author of The Lord of the Rings.',
'Tolkien''s epic high fantasy writing features detailed world-building, invented languages, and mythological depth.',
'International Fantasy Award'),

('C.S.', 'Lewis', NULL, 'cs@lewis.com', '1898-11-29', 'British',
'Clive Staples Lewis was a British writer and lay theologian, best known for The Chronicles of Narnia.',
'Lewis''s writing combines Christian allegory with imaginative fantasy, featuring clear prose and moral themes.',
'Carnegie Medal'),

('Leo', 'Tolstoy', NULL, 'leo@tolstoy.com', '1828-09-09', 'Russian',
'Count Lev Nikolayevich Tolstoy, usually referred to in English as Leo Tolstoy, was a Russian writer regarded as one of the greatest authors of all time.',
'Tolstoy''s realistic fiction is known for detailed character development, moral philosophy, and sweeping historical narratives.',
'Considered master of realistic fiction'),

('Fyodor', 'Dostoevsky', NULL, 'fyodor@dostoevsky.com', '1821-11-11', 'Russian',
'Fyodor Mikhailovich Dostoevsky was a Russian novelist, short story writer, essayist, and journalist.',
'Dostoevsky''s writing explores psychology in troubled political, social, and spiritual contexts of 19th-century Russia.',
'Regarded as one of the greatest psychologists in world literature'),

('Charlotte', 'Brontë', NULL, 'charlotte@bronte.com', '1816-04-21', 'British',
'Charlotte Brontë was an English novelist and poet, the eldest of the three Brontë sisters who survived into adulthood.',
'Brontë''s writing features strong, complex female protagonists and explores themes of morality, social criticism, and romance.',
'Pioneer of literary feminism');


-- Insert 20 Publishers
INSERT INTO publishers (publisher_name, legal_name, email, phone, website_url, address_line1, city, country, company_description, company_description_embed, publishing_focus_description, publishing_focus_description_embed, year_founded) VALUES
('Penguin Random House', 'Penguin Random House LLC', 'contact@penguinrandomhouse.com', '+1-212-782-9000', 'https://www.penguinrandomhouse.com',
'1745 Broadway', 'New York', 'USA',
'Penguin Random House is the world''s largest trade book publisher with a mission to ignite a universal passion for reading by creating books for everyone.', NULL,
'The publisher focuses on literary fiction, commercial fiction, narrative non-fiction, memoir, and children''s books across all genres.', NULL,
1927),

('HarperCollins', 'HarperCollins Publishers LLC', 'info@harpercollins.com', '+1-212-207-7000', 'https://www.harpercollins.com',
'195 Broadway', 'New York', 'USA',
'HarperCollins is one of the world''s foremost publishing companies with a rich history and strong global presence.', NULL,
'Specializes in literary and commercial fiction, business books, cookbooks, mystery, romance, reference, religious, and spiritual books.', NULL,
1989),

('Simon & Schuster', 'Simon & Schuster Inc', 'contact@simonandschuster.com', '+1-212-698-7000', 'https://www.simonandschuster.com',
'1230 Avenue of the Americas', 'New York', 'USA',
'Simon & Schuster is one of the largest English-language publishers, known for quality fiction and nonfiction.', NULL,
'Focuses on commercial and literary fiction, biography, history, self-help, and children''s books.', NULL,
1924),

('Macmillan Publishers', 'Macmillan Publishers Ltd', 'info@macmillan.com', '+44-20-7833-4000', 'https://www.macmillan.com',
'The Macmillan Campus', 'London', 'United Kingdom',
'Macmillan Publishers is a global publishing company with imprints around the world publishing a broad range of works.', NULL,
'Publishes fiction, nonfiction, children''s books, and academic works across diverse subjects and genres.', NULL,
1843),

('Hachette Book Group', 'Hachette Book Group Inc', 'info@hbgusa.com', '+1-212-364-1100', 'https://www.hachettebookgroup.com',
'1290 Avenue of the Americas', 'New York', 'USA',
'Hachette Book Group is a leading trade publisher based in New York and a division of Hachette Livre.', NULL,
'Known for publishing award-winning fiction, nonfiction, business, and children''s books.', NULL,
1837),

('Oxford University Press', 'Oxford University Press', 'enquiry@oup.com', '+44-1865-556767', 'https://www.oup.com',
'Great Clarendon Street', 'Oxford', 'United Kingdom',
'Oxford University Press is the largest university press in the world, furthering the University''s objective of excellence in research.', NULL,
'Focuses on academic, educational, and research publications across all disciplines, particularly dictionaries and reference works.', NULL,
1586),

('Cambridge University Press', 'Cambridge University Press', 'information@cambridge.org', '+44-1223-358331', 'https://www.cambridge.org',
'University Printing House', 'Cambridge', 'United Kingdom',
'Cambridge University Press is part of the University of Cambridge, bringing knowledge, learning and research to the world.', NULL,
'Publishes academic books, journals, and educational materials across science, technology, medicine, and humanities.', NULL,
1534),

('Scholastic Corporation', 'Scholastic Corporation', 'news@scholastic.com', '+1-212-343-6100', 'https://www.scholastic.com',
'557 Broadway', 'New York', 'USA',
'Scholastic is the world''s largest publisher and distributor of children''s books, educational materials, and entertainment.', NULL,
'Specializes in children''s and young adult literature, educational publishing, and literacy programs.', NULL,
1920),

('Bloomsbury Publishing', 'Bloomsbury Publishing Plc', 'contact@bloomsbury.com', '+44-20-7631-5600', 'https://www.bloomsbury.com',
'50 Bedford Square', 'London', 'United Kingdom',
'Bloomsbury is an independent publishing house known for literary quality and commercial success worldwide.', NULL,
'Publishes literary fiction, nonfiction, children''s books, and academic works with focus on quality and discovery.', NULL,
1986),

('Vintage Books', 'Vintage Books (Random House)', 'vintage@penguinrandomhouse.com', '+1-212-572-2420', 'https://www.vintagebooks.com',
'1745 Broadway', 'New York', 'USA',
'Vintage Books publishes a wide range of fiction and nonfiction, bringing exceptional works to mass market audiences.', NULL,
'Known for trade paperback editions of quality fiction and nonfiction, including classics and contemporary works.', NULL,
1954),

('Tor Books', 'Tom Doherty Associates LLC', 'publicity@tor.com', '+1-212-388-0100', 'https://www.tor.com',
'120 Broadway', 'New York', 'USA',
'Tor Books is the largest publisher of science fiction and fantasy in the United States.', NULL,
'Exclusively focuses on science fiction, fantasy, and horror literature, publishing hardcovers, trade paperbacks, and mass market.', NULL,
1980),

('Del Rey Books', 'Del Rey Books (Random House)', 'delreyinfo@penguinrandomhouse.com', '+1-212-782-9000', 'https://www.delreybooks.com',
'1745 Broadway', 'New York', 'USA',
'Del Rey is a premier publisher of science fiction, fantasy, and speculative fiction.', NULL,
'Specializes in science fiction, fantasy, alternate history, and manga with both new voices and established authors.', NULL,
1977),

('Doubleday', 'Doubleday (Penguin Random House)', 'doubleday@penguinrandomhouse.com', '+1-212-782-9000', 'https://www.doubleday.com',
'1745 Broadway', 'New York', 'USA',
'Doubleday has a long tradition of publishing quality fiction and nonfiction for discerning readers.', NULL,
'Publishes literary and commercial fiction, narrative nonfiction, and serious works across various genres.', NULL,
1897),

('Alfred A. Knopf', 'Alfred A. Knopf (Random House)', 'knopf@penguinrandomhouse.com', '+1-212-751-2600', 'https://www.knopf.com',
'1745 Broadway', 'New York', 'USA',
'Knopf is known for publishing distinguished fiction, nonfiction, and poetry from around the world.', NULL,
'Focuses on literary fiction, serious nonfiction, poetry, and works of enduring quality and importance.', NULL,
1915),

('Farrar, Straus and Giroux', 'Farrar, Straus and Giroux LLC', 'fsg.editorial@fsgbooks.com', '+1-212-741-6900', 'https://www.fsgbooks.com',
'120 Broadway', 'New York', 'USA',
'FSG is an American book publishing company known for publishing literary books and poetry.', NULL,
'Renowned for literary fiction, nonfiction, and poetry with emphasis on quality and artistic merit.', NULL,
1946),

('Grove Atlantic', 'Grove Atlantic Inc', 'info@groveatlantic.com', '+1-212-614-7850', 'https://www.groveatlantic.com',
'154 West 14th Street', 'New York', 'USA',
'Grove Atlantic is an independent literary publisher known for publishing innovative and provocative literature.', NULL,
'Publishes literary fiction, nonfiction, poetry, and drama with focus on international and avant-garde works.', NULL,
1917),

('Little, Brown and Company', 'Little, Brown and Company', 'publicity@lbschool.com', '+1-212-364-1100', 'https://www.littlebrown.com',
'1290 Avenue of the Americas', 'New York', 'USA',
'Little, Brown and Company is a publisher committed to publishing books of quality and lasting value.', NULL,
'Publishes fiction, nonfiction, and children''s books with emphasis on literary quality and commercial appeal.', NULL,
1837),

('Viking Press', 'Viking Press (Penguin)', 'viking@penguinrandomhouse.com', '+1-212-366-2000', 'https://www.viking.com',
'1745 Broadway', 'New York', 'USA',
'Viking Press publishes quality fiction and nonfiction for adult readers seeking literary excellence.', NULL,
'Known for publishing literary fiction, narrative nonfiction, biography, and works of cultural significance.', NULL,
1925),

('Vintage Anchor Publishing', 'Vintage Anchor Publishing', 'vintageanchor@penguinrandomhouse.com', '+1-212-572-2420', 'https://www.vintageanchor.com',
'1745 Broadway', 'New York', 'USA',
'Vintage Anchor Publishing brings paperback editions of quality fiction and nonfiction to wide audiences.', NULL,
'Focuses on literary and commercial fiction, nonfiction, and translations in affordable paperback formats.', NULL,
1990),

('Crown Publishing Group', 'Crown Publishing Group', 'crownpublicity@penguinrandomhouse.com', '+1-212-782-9000', 'https://www.crownpublishing.com',
'1745 Broadway', 'New York', 'USA',
'Crown Publishing Group publishes bestselling fiction and nonfiction across multiple imprints.', NULL,
'Known for commercial and literary fiction, memoirs, cookbooks, self-help, and narrative nonfiction.', NULL,
1933);


-- Insert 15 Categories
INSERT INTO categories (category_name, category_code, category_description, category_description_embed, typical_content_description, typical_content_description_embed, age_group, display_order) VALUES
('Fiction', 'FIC', 
'Fiction encompasses works of imaginative narration, especially in prose form. These are stories created from the imagination, not presented as fact, though they may be based on true events or people.', NULL,
'Fiction includes novels and short stories across all genres including literary fiction, contemporary fiction, and general fiction with character development, plot, setting, and theme.', NULL,
'Adult', 1),

('Science Fiction', 'SCI-FI', 
'Science fiction is a genre of speculative fiction dealing with imaginative concepts such as futuristic science and technology, space exploration, time travel, parallel universes, and extraterrestrial life.', NULL,
'Science fiction books typically explore the impact of actual or imagined science on society or individuals, featuring advanced technology, space travel, AI, and speculative futures.', NULL,
'Young Adult/Adult', 2),

('Fantasy', 'FANTASY', 
'Fantasy is a genre of speculative fiction set in a fictional universe, often inspired by real world myths and folklore. It typically features magical elements, mythical creatures, and epic quests.', NULL,
'Fantasy literature includes high fantasy with complex world-building, urban fantasy set in contemporary settings, and magical realism blending fantasy with reality.', NULL,
'All Ages', 3),

('Mystery', 'MYSTERY', 
'Mystery fiction is a genre of fiction that follows a crime, usually a murder, from the moment it is committed to the moment it is solved. It focuses on the investigation and puzzle-solving aspects.', NULL,
'Mystery books feature detective work, clues, red herrings, and logical deduction, often with amateur detectives, police procedurals, or private investigators.', NULL,
'Adult', 4),

('Thriller', 'THRILLER', 
'Thriller is a genre of literature characterized by excitement, suspense, and anticipation. Thrillers keep readers on the edge of their seats with high stakes and constant danger.', NULL,
'Thriller novels feature fast-paced plots, danger, high stakes, and tension, including psychological thrillers, spy thrillers, and action thrillers.', NULL,
'Adult', 5),

('Romance', 'ROM', 
'Romance is a genre of fiction that focuses on the relationship and romantic love between two people, typically with an emotionally satisfying and optimistic ending.', NULL,
'Romance novels center on love stories with emotionally satisfying endings, exploring themes of attraction, relationships, and commitment across various settings.', NULL,
'Adult', 6),

('Horror', 'HORROR', 
'Horror is a genre of fiction intended to frighten, scare, or disgust readers. It often features supernatural elements, psychological terror, or graphic violence.', NULL,
'Horror literature creates fear through supernatural elements, psychological terror, monsters, ghosts, or disturbing scenarios that evoke dread and suspense.', NULL,
'Adult', 7),

('Classic Literature', 'CLASSIC', 
'Classic literature refers to works of fiction and non-fiction considered to be of the highest quality and lasting value, having stood the test of time.', NULL,
'Classic literature includes timeless works exploring universal themes of human nature, society, and existence with complex characters and enduring relevance.', NULL,
'Adult', 8),

('Young Adult', 'YA', 
'Young Adult literature is fiction written for readers aged 12 to 18, featuring teenage protagonists and addressing themes relevant to adolescents.', NULL,
'YA books explore coming-of-age themes, identity, relationships, and challenges faced by teenagers, often featuring first love, self-discovery, and growth.', NULL,
'Young Adult', 9),

('Children''s Literature', 'CHILDREN', 
'Children''s literature includes books written and published for young readers, from picture books to chapter books and middle grade fiction.', NULL,
'Children''s books feature age-appropriate content, often with illustrations, focusing on adventure, friendship, family, and moral lessons.', NULL,
'Children', 10),

('Non-Fiction', 'NON-FIC', 
'Non-fiction is prose writing based on facts, real events, and real people, including biography, history, science, essays, and self-help.', NULL,
'Non-fiction provides factual information, analysis, or arguments about real-world subjects, aiming to inform, explain, or persuade readers.', NULL,
'All Ages', 11),

('Biography/Memoir', 'BIO', 
'Biography is a detailed description of a person''s life written by someone else, while memoir is a personal account of the author''s own life experiences.', NULL,
'Biography and memoir books explore real people''s lives, achievements, struggles, and experiences, providing insight into historical figures or personal journeys.', NULL,
'Adult', 12),

('Historical Fiction', 'HIST-FIC', 
'Historical fiction is a literary genre in which the plot takes place in a setting located in the past, often with real historical events and figures.', NULL,
'Historical fiction blends real historical events, periods, and settings with fictional characters and narratives, bringing the past to life.', NULL,
'Adult', 13),

('Poetry', 'POETRY', 
'Poetry is a form of literature that uses aesthetic and rhythmic qualities of language to evoke meanings in addition to or in place of prosaic ostensible meaning.', NULL,
'Poetry collections feature verse in various forms including sonnets, free verse, haiku, and narrative poetry, expressing emotions and ideas through condensed language.', NULL,
'All Ages', 14),

('Graphic Novels', 'GRAPHIC', 
'Graphic novels are book-length narratives told through a combination of text and sequential art, often in comic book style but with more complex themes.', NULL,
'Graphic novels use visual storytelling combined with text to tell complete narratives, spanning all genres from superhero stories to literary fiction.', NULL,
'All Ages', 15);


-- ============================================================================
-- SAMPLE BOOKS INSERTION
-- ============================================================================

-- Insert 20 Books
INSERT INTO books (isbn_10, isbn_13, title, subtitle, author_id, publisher_id, category_id, publication_date, edition, language, page_count, dimensions, weight_grams, book_description, book_description_embed, detailed_summary, detailed_summary_embed, retail_price, cost_price, discount_percentage, table_of_contents, table_of_contents_embed, total_sales, is_bestseller) VALUES

('0141036144', '9780141036144', '1984', 'A Novel', 1, 1, 1, '1949-06-08', '1st Edition', 'English', 328, '7.8 x 5.1 x 0.9 inches', 250, 'A dystopian social science fiction novel and cautionary tale about the dangers of totalitarianism.', NULL, 'In a world of perpetual war, omnipresent government surveillance, and public manipulation, Winston Smith works for the Ministry of Truth, rewriting history to match the ever-changing party line. When he begins a forbidden love affair with Julia, he discovers the true cost of freedom and the power of the human spirit.', NULL, 12.99, 8.50, 0.00, 'Part I: Chapters 1-8, Part II: Chapters 1-10, Part III: Chapters 1-6, Appendix: The Principles of Newspeak', NULL, 1500000, TRUE),

('0141439510', '9780141439518', 'Pride and Prejudice', NULL, 2, 2, 8, '1813-01-28', '1st Edition', 'English', 432, '8.0 x 5.2 x 1.0 inches', 320, 'A romantic novel of manners written by Jane Austen, following the character development of Elizabeth Bennet.', NULL, 'The story follows Elizabeth Bennet, a spirited and intelligent young woman, as she navigates the social expectations of Regency England. When the proud Mr. Darcy enters her life, their initial mutual dislike gradually transforms into understanding and love, but not without overcoming pride, prejudice, and social obstacles.', NULL, 9.99, 6.75, 15.00, 'Volume I: Chapters 1-23, Volume II: Chapters 1-19, Volume III: Chapters 1-19', NULL, 2000000, TRUE),

('0061120081', '9780061120081', 'One Hundred Years of Solitude', NULL, 3, 3, 1, '1967-05-30', '1st Edition', 'Spanish', 417, '8.0 x 5.3 x 1.2 inches', 380, 'A landmark novel that tells the multi-generational story of the Buendía family and the mythical town of Macondo.', NULL, 'This masterpiece of magical realism chronicles seven generations of the Buendía family, whose patriarch, José Arcadio Buendía, founds the town of Macondo. The novel explores themes of solitude, love, death, and the cyclical nature of history through a rich tapestry of magical and realistic elements.', NULL, 15.99, 10.25, 0.00, 'Part I: The Buendía Family Origins, Part II: The Civil Wars, Part III: The Banana Company, Part IV: The Final Generation', NULL, 500000, TRUE),

('0307743659', '9780307743657', 'The Shining', NULL, 4, 4, 7, '1977-01-28', '1st Edition', 'English', 688, '8.2 x 5.5 x 1.4 inches', 520, 'A horror novel about a writer who becomes the winter caretaker of an isolated hotel.', NULL, 'Jack Torrance, a struggling writer and recovering alcoholic, takes a job as the winter caretaker of the Overlook Hotel in Colorado. As the hotel''s supernatural forces begin to influence him, his young son Danny, who possesses psychic abilities, must fight to save his family from the hotel''s malevolent spirits.', NULL, 16.99, 11.50, 10.00, 'Part I: Before the Play, Part II: The Play, Part III: After the Play, Epilogue', NULL, 800000, TRUE),

('0439708184', '9780439708180', 'Harry Potter and the Philosopher''s Stone', NULL, 5, 5, 3, '1997-06-26', '1st Edition', 'English', 223, '7.6 x 5.0 x 1.0 inches', 200, 'The first novel in the Harry Potter series, following a young wizard''s first year at Hogwarts School of Witchcraft and Wizardry.', NULL, 'Harry Potter, an orphaned boy living with his cruel aunt and uncle, discovers on his eleventh birthday that he is a wizard. He is whisked away to Hogwarts School of Witchcraft and Wizardry, where he makes friends, learns magic, and uncovers the truth about his parents'' death and his connection to the dark wizard Voldemort.', NULL, 12.99, 8.25, 0.00, 'Chapter 1: The Boy Who Lived through Chapter 17: The Man with Two Faces', NULL, 120000000, TRUE),

('0062073488', '9780062073488', 'Murder on the Orient Express', NULL, 6, 6, 4, '1934-01-01', '1st Edition', 'English', 288, '7.8 x 5.1 x 0.8 inches', 220, 'A detective novel featuring Hercule Poirot investigating a murder aboard the famous train.', NULL, 'When the luxurious Orient Express is stopped by heavy snowfall, one of its passengers is found murdered in his locked compartment. The brilliant detective Hercule Poirot must solve the case before the train can continue its journey, uncovering a complex web of motives and alibis among the diverse group of passengers.', NULL, 13.99, 9.00, 5.00, 'Part I: The Facts, Part II: The Evidence, Part III: Hercule Poirot Sits Back and Thinks', NULL, 1000000, TRUE),

('0684801469', '9780684801469', 'The Old Man and the Sea', NULL, 7, 7, 1, '1952-09-01', '1st Edition', 'English', 127, '8.0 x 5.3 x 0.4 inches', 120, 'A short novel about an aging Cuban fisherman and his struggle with a giant marlin.', NULL, 'Santiago, an old Cuban fisherman, has gone eighty-four days without catching a fish. When he finally hooks a massive marlin, he engages in an epic three-day battle with the fish, testing his strength, courage, and determination in a story that explores themes of perseverance, dignity, and man''s relationship with nature.', NULL, 11.99, 7.50, 0.00, 'A single continuous narrative divided into three days of struggle', NULL, 2000000, TRUE),

('1400033411', '9781400033413', 'Beloved', NULL, 8, 8, 1, '1987-09-02', '1st Edition', 'English', 324, '8.0 x 5.2 x 0.8 inches', 280, 'A novel about a former slave and the haunting presence of her deceased daughter.', NULL, 'Sethe, a former slave living in post-Civil War Ohio, is haunted by the ghost of her baby daughter, whom she killed to save from slavery. When a mysterious young woman named Beloved appears, Sethe must confront her traumatic past and the devastating legacy of slavery in this powerful exploration of memory, trauma, and the bonds of family.', NULL, 14.99, 9.75, 0.00, 'Part I: 124 was spiteful, Part II: 124 was loud, Part III: 124 was quiet', NULL, 800000, TRUE),

('0375704027', '9780375704024', 'Norwegian Wood', NULL, 9, 9, 1, '1987-09-04', '1st Edition', 'Japanese', 296, '8.0 x 5.2 x 0.8 inches', 250, 'A coming-of-age novel about love, loss, and the transition to adulthood in 1960s Tokyo.', NULL, 'Toru Watanabe, a college student in Tokyo, finds himself caught between two women: Naoko, the girlfriend of his best friend who committed suicide, and Midori, a vibrant and independent classmate. This melancholic tale explores themes of love, death, mental illness, and the search for meaning in a rapidly changing world.', NULL, 15.99, 10.50, 0.00, 'Chapters 1-11 following Toru''s relationships and personal growth', NULL, 3000000, TRUE),

('0385490819', '9780385490818', 'The Handmaid''s Tale', NULL, 10, 10, 1, '1985-04-01', '1st Edition', 'English', 311, '8.0 x 5.2 x 0.8 inches', 260, 'A dystopian novel set in a totalitarian society where women are subjugated and used for reproduction.', NULL, 'In the Republic of Gilead, a theocratic dictatorship has replaced the United States. Offred, a Handmaid, is forced to bear children for the ruling class. Through her first-person narrative, she reveals the horrors of this society while secretly recording her story, hoping it will survive for future generations to understand what happened.', NULL, 16.99, 11.00, 0.00, 'Part I: Night, Part II: Shopping, Part III: Night, Part IV: Nap, Part V: Nap, Part VI: Household, Part VII: Night, Part VIII: Birth Day, Part IX: Night, Part X: Soul Scrolls, Part XI: Night, Part XII: Jezebel''s, Part XIII: Night, Part XIV: Salvaging, Part XV: Night, Historical Notes', NULL, 2000000, TRUE),

('0062255655', '9780062255655', 'American Gods', NULL, 11, 11, 3, '2001-06-19', '1st Edition', 'English', 635, '8.2 x 5.5 x 1.6 inches', 580, 'A fantasy novel about old gods struggling to survive in modern America.', NULL, 'Shadow Moon, recently released from prison, finds himself caught in a war between the old gods of mythology and the new gods of technology and media. As he travels across America with the mysterious Mr. Wednesday, he discovers a hidden world where ancient deities walk among mortals, fighting for relevance in a changing world.', NULL, 17.99, 12.25, 0.00, 'Part I: Shadow, Part II: My Ainsel, Part III: The Moment of the Storm, Part IV: The Moment of the Storm (continued)', NULL, 1500000, TRUE),

('0307277712', '9780307277712', 'Half of a Yellow Sun', NULL, 12, 12, 13, '2006-08-01', '1st Edition', 'English', 433, '8.0 x 5.2 x 1.0 inches', 350, 'A novel set during the Nigerian Civil War, exploring the lives of characters caught in the conflict.', NULL, 'Through the intertwined stories of Ugwu, a houseboy; Olanna, a university lecturer; and Richard, a British expatriate, this novel brings to life the Biafran War of 1967-1970. The narrative explores themes of love, loyalty, betrayal, and the devastating impact of war on ordinary people''s lives.', NULL, 15.99, 10.75, 0.00, 'Part I: The Early Sixties, Part II: The Late Sixties, Part III: The Early Sixties, Part IV: The Late Sixties', NULL, 600000, TRUE),

('0307277674', '9780307277675', 'Never Let Me Go', NULL, 13, 13, 1, '2005-03-01', '1st Edition', 'English', 288, '8.0 x 5.2 x 0.8 inches', 240, 'A dystopian novel about clones raised to be organ donors in an alternate version of England.', NULL, 'Kathy H., a thirty-one-year-old carer, reflects on her time at Hailsham, an exclusive boarding school where she and her friends Ruth and Tommy were raised. As she recounts their relationships and the truth about their purpose in life, the novel explores themes of mortality, love, and what it means to be human.', NULL, 14.99, 9.50, 0.00, 'Part I: Hailsham, Part II: The Cottages, Part III: Norfolk', NULL, 800000, TRUE),

('0061122416', '9780061122415', 'The Alchemist', NULL, 14, 14, 1, '1988-01-01', '1st Edition', 'Portuguese', 163, '7.8 x 5.1 x 0.5 inches', 140, 'A philosophical novel about a young Andalusian shepherd''s journey to find a worldly treasure.', NULL, 'Santiago, a young Andalusian shepherd, dreams of finding a treasure in the Egyptian pyramids. His journey takes him across the Mediterranean and the Sahara Desert, where he learns about the importance of following one''s dreams, listening to one''s heart, and recognizing life''s omens in this allegorical tale of self-discovery.', NULL, 13.99, 9.25, 0.00, 'Part One: The Shepherd''s Journey, Part Two: The Alchemist''s Wisdom', NULL, 65000000, TRUE),

('0156030415', '9780156030410', 'Mrs. Dalloway', NULL, 15, 15, 8, '1925-05-14', '1st Edition', 'English', 194, '8.0 x 5.2 x 0.6 inches', 160, 'A modernist novel following a day in the life of Clarissa Dalloway in post-World War I London.', NULL, 'On a single day in June 1923, Clarissa Dalloway prepares for a party she will host that evening. Through stream-of-consciousness narration, the novel explores her thoughts, memories, and relationships, while also following the parallel story of Septimus Warren Smith, a shell-shocked veteran, in this meditation on time, memory, and the human experience.', NULL, 12.99, 8.00, 0.00, 'A single day''s narrative from morning to evening party', NULL, 500000, TRUE),

('1451673310', '9781451673319', 'Fahrenheit 451', NULL, 16, 16, 2, '1953-10-19', '1st Edition', 'English', 249, '8.0 x 5.2 x 0.7 inches', 200, 'A dystopian novel about a future society where books are banned and burned by firemen.', NULL, 'Guy Montag is a fireman whose job is to burn books in a society where reading is forbidden. When he meets a young woman who challenges his worldview, he begins to question everything he has been taught. This classic novel explores themes of censorship, knowledge, and the power of literature to transform society.', NULL, 13.99, 9.00, 0.00, 'Part I: The Hearth and the Salamander, Part II: The Sieve and the Sand, Part III: Burning Bright', NULL, 10000000, TRUE),

('0441013597', '9780441013593', 'The Left Hand of Darkness', NULL, 17, 17, 2, '1969-03-01', '1st Edition', 'English', 304, '8.0 x 5.2 x 0.8 inches', 250, 'A science fiction novel about a human envoy on a planet where inhabitants can change gender.', NULL, 'Genly Ai, an envoy from the Ekumen, travels to the planet Gethen to convince its inhabitants to join an interplanetary alliance. On this world where people are neither male nor female but can become either during their monthly cycle, Ai must navigate complex political and cultural differences while exploring themes of gender, identity, and communication.', NULL, 15.99, 10.50, 0.00, 'Chapters 1-20 following Genly Ai''s mission and experiences', NULL, 400000, TRUE),

('0099587124', '9780099587120', 'Midnight''s Children', NULL, 18, 18, 13, '1981-04-01', '1st Edition', 'English', 647, '8.0 x 5.2 x 1.4 inches', 520, 'A magical realist novel about children born at the moment of India''s independence.', NULL, 'Saleem Sinai, born at the exact moment of India''s independence, discovers he has telepathic powers that connect him to other children born in the same hour. His life story becomes intertwined with the history of modern India, exploring themes of identity, nationalism, and the relationship between personal and political history.', NULL, 16.99, 11.25, 0.00, 'Book I: The Perforated Sheet, Book II: Mercurochrome, Book III: The Buddha', NULL, 300000, TRUE),

('0395927204', '9780395927209', 'Interpreter of Maladies', NULL, 19, 19, 1, '1999-06-01', '1st Edition', 'English', 198, '8.0 x 5.2 x 0.6 inches', 160, 'A collection of short stories exploring the Indian American experience.', NULL, 'This Pulitzer Prize-winning collection of nine short stories explores the lives of Indian Americans, both in India and the United States. Through themes of cultural displacement, family relationships, and the immigrant experience, Lahiri creates intimate portraits of characters navigating between two worlds and cultures.', NULL, 14.99, 9.75, 0.00, 'A Temporary Matter, When Mr. Pirzada Came to Dine, Interpreter of Maladies, A Real Durwan, Sexy, Mrs. Sen''s, This Blessed House, The Treatment of Bibi Haldar, The Third and Final Continent', NULL, 200000, TRUE),

('0385542364', '9780385542361', 'The Underground Railroad', NULL, 20, 20, 13, '2016-08-02', '1st Edition', 'English', 306, '8.2 x 5.5 x 0.9 inches', 280, 'A novel reimagining the Underground Railroad as an actual railroad system.', NULL, 'Cora, a young slave on a Georgia plantation, escapes via the Underground Railroad, which in this alternate history is a literal railroad system. As she travels from state to state, each stop represents a different aspect of American history and the ongoing struggle for freedom, in this powerful exploration of slavery and its legacy.', NULL, 16.99, 11.50, 0.00, 'Georgia, South Carolina, North Carolina, Tennessee, Indiana, The North', NULL, 800000, TRUE);


-- ============================================================================
-- SAMPLE REVIEWS INSERTION
-- ============================================================================

-- Insert 20 Reviews
INSERT INTO reviews (book_id, reviewer_name, reviewer_email, verified_purchase, rating, review_title, review_text, review_text_embed, detailed_feedback, detailed_feedback_embed, review_date, helpful_count, not_helpful_count, is_verified, is_published, moderation_notes, moderation_notes_embed, reading_duration_days, would_recommend, reader_age_group) VALUES

(1, 'Sarah Johnson', 'sarah.johnson@email.com', TRUE, 5, 'Absolutely Chilling and Timeless', 'This book is more relevant today than ever. Orwell''s vision of a surveillance state feels eerily prescient in our digital age. The writing is masterful and the story is both heartbreaking and terrifying.', NULL, 'I first read this in high school and it didn''t fully resonate with me. Re-reading it as an adult, I''m struck by how accurate Orwell''s predictions were about government control, media manipulation, and the erosion of privacy. Winston''s journey from compliance to rebellion is both tragic and inspiring. The ending still haunts me.', NULL, '2023-03-15', 127, 3, TRUE, TRUE, NULL, NULL, 5, TRUE, 'Adult'),

(2, 'Michael Chen', 'm.chen@email.com', TRUE, 5, 'A Perfect Love Story', 'Jane Austen at her finest. The wit, the social commentary, and the character development are all perfect. Elizabeth Bennet remains one of literature''s greatest heroines.', NULL, 'I''ve read this book multiple times and each reading reveals new layers. Austen''s social satire is brilliant, and the romance between Elizabeth and Darcy is beautifully developed. The dialogue is sharp and witty, and the supporting characters are wonderfully drawn. A true masterpiece.', NULL, '2023-01-22', 89, 2, TRUE, TRUE, NULL, NULL, 7, TRUE, 'Adult'),

(3, 'Maria Rodriguez', 'maria.rodriguez@email.com', FALSE, 4, 'Magical Realism at its Best', 'García Márquez creates a world that feels both fantastical and deeply human. The Buendía family saga is epic and moving, though sometimes the magical elements can be confusing.', NULL, 'The prose is absolutely beautiful and the magical realism elements are woven seamlessly into the narrative. I found myself getting lost in the family tree at times, but the emotional core of the story is powerful. The themes of solitude and the cyclical nature of history are profound.', NULL, '2023-02-10', 156, 8, FALSE, TRUE, NULL, NULL, 12, TRUE, 'Adult'),

(4, 'David Thompson', 'david.t@email.com', TRUE, 5, 'The Master of Horror', 'Stephen King''s ability to create psychological terror is unmatched. The Shining is not just scary - it''s a deep exploration of addiction, family dynamics, and the power of evil places.', NULL, 'This book terrified me in the best way possible. King''s character development is incredible, especially Jack''s descent into madness. The hotel itself becomes a character, and the supernatural elements feel grounded in psychological reality. The relationship between Jack and Danny is heartbreaking.', NULL, '2023-04-05', 203, 5, TRUE, TRUE, NULL, NULL, 8, TRUE, 'Adult'),

(5, 'Emma Wilson', 'emma.wilson@email.com', TRUE, 5, 'Pure Magic', 'This book started my love of reading. Rowling creates a world so rich and detailed that you can''t help but be drawn in. Harry''s journey from neglected orphan to hero is beautifully told.', NULL, 'I''ve read this book countless times and it never gets old. The world-building is incredible, the characters are memorable, and the plot is perfectly paced. Rowling balances humor, adventure, and emotion beautifully. This is the book that made me fall in love with fantasy.', NULL, '2023-01-08', 445, 12, TRUE, TRUE, NULL, NULL, 4, TRUE, 'Young Adult'),

(6, 'Robert Kim', 'robert.kim@email.com', TRUE, 4, 'Classic Mystery', 'Agatha Christie''s plotting is masterful. The solution is both surprising and fair, and Poirot is a fascinating detective. A must-read for mystery fans.', NULL, 'The locked-room mystery setup is perfect, and Christie''s character development is excellent. Each passenger has clear motives and opportunities, making the deduction process engaging. Poirot''s methodical approach to solving the crime is fascinating to follow.', NULL, '2023-03-28', 78, 4, TRUE, TRUE, NULL, NULL, 6, TRUE, 'Adult'),

(7, 'Lisa Anderson', 'lisa.anderson@email.com', TRUE, 5, 'A Masterpiece of Simplicity', 'Hemingway''s sparse prose is perfect for this story. Every word counts, and the emotional impact is profound. Santiago''s struggle is both physical and spiritual.', NULL, 'This is Hemingway at his best. The story is deceptively simple but contains deep themes about perseverance, dignity, and man''s relationship with nature. Santiago''s character is beautifully drawn, and the fishing scenes are incredibly vivid. The ending is both triumphant and tragic.', NULL, '2023-02-14', 92, 1, TRUE, TRUE, NULL, NULL, 3, TRUE, 'Adult'),

(8, 'James Brown', 'james.brown@email.com', TRUE, 5, 'Powerful and Haunting', 'Toni Morrison''s writing is absolutely stunning. This book deals with difficult themes but does so with incredible grace and power. A true masterpiece of American literature.', NULL, 'Morrison''s prose is poetic and powerful, and her exploration of trauma, memory, and the legacy of slavery is profound. The supernatural elements are handled beautifully, and the characters are complex and fully realized. This is a book that stays with you long after you finish it.', NULL, '2023-01-30', 167, 6, TRUE, TRUE, NULL, NULL, 9, TRUE, 'Adult'),

(9, 'Yuki Tanaka', 'yuki.tanaka@email.com', TRUE, 4, 'Beautifully Melancholic', 'Murakami''s writing style is unique and captivating. The story is sad but beautiful, exploring themes of love, loss, and growing up in 1960s Japan.', NULL, 'The atmosphere of this book is incredible - it perfectly captures the feeling of being young and lost in a changing world. The characters are well-drawn, and the themes of love and loss are handled with sensitivity. Some of the magical realism elements felt a bit forced, but overall it''s a beautiful book.', NULL, '2023-03-12', 134, 7, TRUE, TRUE, NULL, NULL, 6, TRUE, 'Adult'),

(10, 'Jennifer Davis', 'jennifer.davis@email.com', TRUE, 5, 'Disturbingly Relevant', 'Atwood''s dystopian vision feels more relevant than ever. The writing is powerful and the world-building is incredibly detailed. A chilling look at what could be.', NULL, 'This book scared me more than any horror novel because it feels so possible. Atwood''s world-building is meticulous, and her exploration of gender, power, and control is profound. Offred''s voice is compelling, and the story is both intimate and epic in scope.', NULL, '2023-04-18', 189, 4, TRUE, TRUE, NULL, NULL, 7, TRUE, 'Adult'),

(11, 'Alex Morgan', 'alex.morgan@email.com', TRUE, 4, 'Epic Fantasy Adventure', 'Gaiman''s imagination is incredible. The concept of old gods vs. new gods is brilliant, and the execution is mostly successful. Shadow is a great protagonist.', NULL, 'The world-building is fantastic, and Gaiman''s blend of mythology and modern America is creative. The pacing can be slow at times, but the payoff is worth it. The characters are memorable, and the themes about belief and change are thought-provoking.', NULL, '2023-02-25', 112, 9, TRUE, TRUE, NULL, NULL, 10, TRUE, 'Adult'),

(12, 'Fatima Al-Zahra', 'fatima.alzahra@email.com', TRUE, 5, 'Heartbreaking and Beautiful', 'Adichie''s writing is powerful and her characters are unforgettable. This book brings the Biafran War to life in a way that''s both educational and deeply moving.', NULL, 'The multiple perspectives work beautifully, and Adichie''s character development is excellent. The historical context is well-researched, and the personal stories make the larger political events feel immediate and real. This is an important book that everyone should read.', NULL, '2023-01-15', 98, 3, TRUE, TRUE, NULL, NULL, 8, TRUE, 'Adult'),

(13, 'Thomas Lee', 'thomas.lee@email.com', TRUE, 5, 'Subtle and Profound', 'Ishiguro''s writing is understated but incredibly powerful. The story builds slowly but the emotional impact is devastating. A meditation on what it means to be human.', NULL, 'This book is a masterpiece of subtle storytelling. Ishiguro reveals the truth gradually, and the impact is all the more powerful for it. The themes of mortality, love, and acceptance are handled with incredible sensitivity. Kathy''s voice is perfect - restrained but deeply emotional.', NULL, '2023-03-20', 145, 2, TRUE, TRUE, NULL, NULL, 5, TRUE, 'Adult'),

(14, 'Sofia Petrov', 'sofia.petrov@email.com', TRUE, 3, 'Inspiring but Repetitive', 'The message is beautiful and the story is engaging, but Coelho''s writing style can be repetitive. Still, it''s a book that many people find life-changing.', NULL, 'I understand why this book is so popular - the message about following your dreams is universal and inspiring. However, I found the writing style a bit too simplistic and repetitive. The allegorical elements are heavy-handed at times, but the core story is engaging.', NULL, '2023-02-08', 67, 15, TRUE, TRUE, NULL, NULL, 4, FALSE, 'Adult'),

(15, 'William Taylor', 'william.taylor@email.com', TRUE, 4, 'Stream of Consciousness Mastery', 'Woolf''s experimental style is challenging but rewarding. The way she captures the flow of thoughts and memories is incredible. A landmark of modernist literature.', NULL, 'This book requires patience and attention, but the payoff is immense. Woolf''s stream-of-consciousness technique is masterful, and her exploration of time, memory, and social class is profound. The parallel stories of Clarissa and Septimus work beautifully together.', NULL, '2023-04-02', 89, 11, TRUE, TRUE, NULL, NULL, 6, TRUE, 'Adult'),

(16, 'Rachel Green', 'rachel.green@email.com', TRUE, 5, 'Timeless Warning', 'Bradbury''s vision of a book-burning society is chilling and relevant. The writing is beautiful and the message about the importance of literature is powerful.', NULL, 'This book is more relevant than ever in our digital age. Bradbury''s prose is lyrical, and his world-building is detailed and believable. Montag''s transformation from book-burner to book-lover is compelling, and the supporting characters are memorable. A true classic.', NULL, '2023-01-25', 178, 5, TRUE, TRUE, NULL, NULL, 5, TRUE, 'Adult'),

(17, 'Kevin O''Connor', 'kevin.oconnor@email.com', TRUE, 4, 'Thought-Provoking Sci-Fi', 'Le Guin''s exploration of gender and society is fascinating. The world-building is excellent, though the pacing can be slow. A classic of science fiction.', NULL, 'The concept of a genderless society is brilliantly explored, and Le Guin''s world-building is detailed and believable. The political intrigue is engaging, and the themes about communication and understanding are profound. Some sections drag, but overall it''s a rewarding read.', NULL, '2023-03-05', 76, 8, TRUE, TRUE, NULL, NULL, 9, TRUE, 'Adult'),

(18, 'Priya Sharma', 'priya.sharma@email.com', TRUE, 5, 'Magical Realism Masterpiece', 'Rushdie''s writing is exuberant and his imagination is boundless. The connection between personal and political history is brilliantly executed. A true epic.', NULL, 'This book is a masterpiece of magical realism and historical fiction. Rushdie''s prose is rich and playful, and his exploration of Indian history through Saleem''s story is brilliant. The magical elements are seamlessly integrated, and the themes of identity and nationalism are profound.', NULL, '2023-02-18', 123, 4, TRUE, TRUE, NULL, NULL, 11, TRUE, 'Adult'),

(19, 'Daniel Kim', 'daniel.kim@email.com', TRUE, 5, 'Perfect Short Stories', 'Lahiri''s writing is precise and her characters are beautifully drawn. Each story is a small masterpiece. A perfect introduction to her work.', NULL, 'These stories are beautifully crafted, and Lahiri''s exploration of the immigrant experience is sensitive and insightful. The prose is elegant, and the characters feel real and relatable. Each story stands alone but together they create a powerful portrait of cultural displacement and adaptation.', NULL, '2023-04-10', 87, 2, TRUE, TRUE, NULL, NULL, 3, TRUE, 'Adult'),

(20, 'Amanda Foster', 'amanda.foster@email.com', TRUE, 5, 'Powerful Historical Fiction', 'Whitehead''s reimagining of the Underground Railroad is brilliant and harrowing. The writing is powerful and the historical elements are well-researched. A must-read.', NULL, 'This book is both a gripping adventure story and a powerful exploration of American history. Whitehead''s prose is beautiful, and his reimagining of the Underground Railroad as a literal railroad is creative and effective. Cora is a compelling protagonist, and the supporting characters are well-drawn.', NULL, '2023-03-30', 156, 3, TRUE, TRUE, NULL, NULL, 7, TRUE, 'Adult');


-- ============================================================================
-- ADDITIONAL 50 BOOKS INSERTION
-- ============================================================================

-- Insert 50 Additional Books
INSERT INTO books (isbn_10, isbn_13, title, subtitle, author_id, publisher_id, category_id, publication_date, edition, language, page_count, dimensions, weight_grams, book_description, book_description_embed, detailed_summary, detailed_summary_embed, retail_price, cost_price, discount_percentage, table_of_contents, table_of_contents_embed, total_sales, is_bestseller) VALUES

('0141187766', '9780141187761', 'Animal Farm', 'A Fairy Story', 1, 1, 1, '1945-08-17', '1st Edition', 'English', 112, '7.8 x 5.1 x 0.4 inches', 100, 'A satirical allegory about farm animals who rebel against their human farmer, hoping to create a society where the animals can be equal, free, and happy.', NULL, 'When the animals of Manor Farm overthrow their human owner, they establish a new society based on equality and cooperation. However, as the pigs take control, the revolution''s ideals are gradually corrupted, leading to a new form of oppression that mirrors the old regime.', NULL, 10.99, 7.25, 0.00, 'Chapter 1: The Rebellion, Chapter 2: The Seven Commandments, Chapter 3: The Windmill, Chapter 4: The Battle, Chapter 5: The Expulsion, Chapter 6: The Betrayal, Chapter 7: The Purge, Chapter 8: The Final Transformation, Chapter 9: The New Tyranny, Chapter 10: The Indistinguishable End', NULL, 2500000, TRUE),

('0141439847', '9780141439847', 'Emma', NULL, 2, 2, 8, '1815-12-23', '1st Edition', 'English', 474, '8.0 x 5.2 x 1.1 inches', 360, 'A comedy of manners about a young woman who fancies herself a matchmaker but learns about love and self-awareness through her own romantic misadventures.', NULL, 'Emma Woodhouse, handsome, clever, and rich, with a comfortable home and happy disposition, seems to unite some of the best blessings of existence. However, her attempts at matchmaking lead to misunderstandings and complications that ultimately teach her valuable lessons about love, friendship, and her own heart.', NULL, 11.99, 8.00, 5.00, 'Volume I: Chapters 1-18, Volume II: Chapters 1-18, Volume III: Chapters 1-19', NULL, 800000, TRUE),

('0060883286', '9780060883287', 'Love in the Time of Cholera', NULL, 3, 3, 1, '1985-01-01', '1st Edition', 'Spanish', 348, '8.0 x 5.3 x 1.0 inches', 320, 'A romantic novel about unrequited love that spans over fifty years, exploring themes of passion, aging, and the persistence of desire.', NULL, 'Florentino Ariza falls in love with Fermina Daza when they are young, but she marries another man. For over fifty years, Florentino waits for his chance to win her back, living a life of numerous affairs while maintaining his devotion to his first love.', NULL, 14.99, 9.75, 0.00, 'Part I: The Beginning, Part II: The Middle Years, Part III: The Reunion, Part IV: The End', NULL, 1200000, TRUE),

('0307743667', '9780307743664', 'It', NULL, 4, 4, 7, '1986-09-15', '1st Edition', 'English', 1138, '8.2 x 5.5 x 2.0 inches', 900, 'A horror novel about a group of children who face a malevolent entity that preys on the children of Derry, Maine, and their return as adults to confront it again.', NULL, 'In the summer of 1958, seven children in Derry, Maine, face an ancient evil that takes the form of a clown named Pennywise. Twenty-seven years later, they return as adults to fulfill a blood oath and destroy the creature once and for all.', NULL, 18.99, 12.50, 0.00, 'Part I: The Shadow Before, Part II: June of 1958, Part III: Grownups, Part IV: July of 1958, Part V: The Ritual of Chüd, Part VI: Out', NULL, 1500000, TRUE),

('0439708192', '9780439708197', 'Harry Potter and the Chamber of Secrets', NULL, 5, 5, 3, '1998-07-02', '1st Edition', 'English', 251, '7.6 x 5.0 x 1.1 inches', 220, 'The second novel in the Harry Potter series, following Harry''s second year at Hogwarts and the discovery of the Chamber of Secrets.', NULL, 'Harry returns to Hogwarts for his second year, but strange things are happening: students are being petrified, and someone has opened the legendary Chamber of Secrets. Harry must uncover the truth about the chamber and the monster within it.', NULL, 12.99, 8.25, 0.00, 'Chapter 1: The Worst Birthday through Chapter 18: Dobby''s Reward', NULL, 77000000, TRUE),

('0062073496', '9780062073495', 'Death on the Nile', NULL, 6, 6, 4, '1937-11-01', '1st Edition', 'English', 288, '7.8 x 5.1 x 0.8 inches', 220, 'A detective novel featuring Hercule Poirot investigating a murder aboard a Nile steamer during a honeymoon cruise.', NULL, 'While on a luxury cruise down the Nile, Poirot witnesses the murder of a beautiful heiress. With a limited number of suspects and no way to leave the boat, Poirot must solve the case before the killer strikes again.', NULL, 13.99, 9.00, 5.00, 'Part I: The Journey, Part II: The Murder, Part III: The Investigation, Part IV: The Solution', NULL, 800000, TRUE),

('0684801477', '9780684801476', 'For Whom the Bell Tolls', NULL, 7, 7, 1, '1940-10-21', '1st Edition', 'English', 471, '8.0 x 5.3 x 1.2 inches', 400, 'A novel set during the Spanish Civil War, following an American dynamiter who falls in love while fighting for the Republican cause.', NULL, 'Robert Jordan, an American fighting for the Republicans in the Spanish Civil War, is assigned to blow up a bridge. As he prepares for the mission, he falls in love with Maria, a young woman who has suffered at the hands of the Fascists.', NULL, 15.99, 10.50, 0.00, 'Part I: The Mission, Part II: The Preparation, Part III: The Attack, Part IV: The Aftermath', NULL, 600000, TRUE),

('1400033429', '9781400033420', 'Song of Solomon', NULL, 8, 8, 1, '1977-08-12', '1st Edition', 'English', 337, '8.0 x 5.2 x 0.9 inches', 290, 'A novel following the life of Macon "Milkman" Dead III, exploring themes of identity, family history, and the African American experience.', NULL, 'Milkman Dead grows up in a wealthy African American family but feels disconnected from his roots. His journey to discover his family history leads him to understand his identity and the legacy of slavery and migration.', NULL, 15.99, 10.25, 0.00, 'Part I: The Present, Part II: The Past, Part III: The Journey, Part IV: The Return', NULL, 400000, TRUE),

('0375704035', '9780375704031', 'Kafka on the Shore', NULL, 9, 9, 1, '2002-09-12', '1st Edition', 'Japanese', 467, '8.0 x 5.2 x 1.1 inches', 400, 'A surreal novel following two parallel stories: a teenager who runs away from home and an elderly man who can talk to cats.', NULL, 'Fifteen-year-old Kafka Tamura runs away from home to escape an Oedipal prophecy, while Nakata, an elderly man who lost his memory in a childhood accident, can communicate with cats. Their paths converge in a story that blends reality with the supernatural.', NULL, 16.99, 11.25, 0.00, 'Chapters 1-49 alternating between Kafka and Nakata''s stories', NULL, 2000000, TRUE),

('0385490827', '9780385490825', 'The Testaments', NULL, 10, 10, 1, '2019-09-10', '1st Edition', 'English', 432, '8.0 x 5.2 x 1.0 inches', 350, 'A sequel to The Handmaid''s Tale, told from the perspectives of three women living in Gilead fifteen years after the original story.', NULL, 'Fifteen years after the events of The Handmaid''s Tale, three women''s lives intersect in the Republic of Gilead: Aunt Lydia, a high-ranking official; Agnes, a young woman raised in Gilead; and Daisy, a teenager living in Canada. Their stories reveal the inner workings of Gilead and the resistance movement.', NULL, 17.99, 12.00, 0.00, 'Part I: The Ardua Hall Holograph, Part II: Transcript of Witness Testimony 369A, Part III: The Ardua Hall Holograph (continued)', NULL, 1000000, TRUE),

('0062255663', '9780062255662', 'Good Omens', 'The Nice and Accurate Prophecies of Agnes Nutter, Witch', 11, 11, 3, '1990-05-01', '1st Edition', 'English', 432, '8.0 x 5.2 x 1.0 inches', 350, 'A comedic fantasy novel about an angel and demon who team up to prevent the apocalypse, written by Neil Gaiman and Terry Pratchett.', NULL, 'Aziraphale, an angel, and Crowley, a demon, have grown fond of Earth and don''t want to see it destroyed in the apocalypse. When the Antichrist is born, they team up to prevent the end of the world, but things don''t go according to plan.', NULL, 15.99, 10.50, 0.00, 'Part I: The Beginning, Part II: The Middle, Part III: The End', NULL, 1800000, TRUE),

('0307277720', '9780307277719', 'Purple Hibiscus', NULL, 12, 12, 1, '2003-10-21', '1st Edition', 'English', 307, '8.0 x 5.2 x 0.8 inches', 250, 'A coming-of-age novel about a young Nigerian girl growing up in a strict Catholic household during political unrest.', NULL, 'Fifteen-year-old Kambili lives in Nigeria under the rule of her fanatically religious father. When she visits her aunt''s house, she discovers a different way of life and begins to question her family''s values and her own identity.', NULL, 14.99, 9.75, 0.00, 'Part I: Breaking Gods, Part II: Speaking with Our Spirits, Part III: The Pieces of Gods', NULL, 500000, TRUE),

('0307277682', '9780307277689', 'The Remains of the Day', NULL, 13, 13, 1, '1989-05-01', '1st Edition', 'English', 245, '8.0 x 5.2 x 0.7 inches', 200, 'A novel about an English butler reflecting on his life and career during a road trip through the English countryside.', NULL, 'Stevens, a butler at Darlington Hall, takes a road trip to visit a former colleague. As he travels, he reflects on his life of service, his relationship with his father, and the missed opportunities for love and personal fulfillment.', NULL, 13.99, 9.00, 0.00, 'Day One: Evening, Day Two: Morning, Day Two: Afternoon, Day Three: Morning, Day Three: Evening, Day Four: Afternoon, Day Six: Evening', NULL, 900000, TRUE),

('0061122424', '9780061122422', 'Eleven Minutes', NULL, 14, 14, 1, '2003-04-01', '1st Edition', 'Portuguese', 273, '7.8 x 5.1 x 0.7 inches', 220, 'A novel about a young Brazilian woman who becomes a prostitute in Switzerland and her journey of self-discovery.', NULL, 'Maria, a young woman from Brazil, travels to Switzerland in search of adventure and money. She becomes a prostitute but gradually discovers the difference between sex and love, leading to a profound transformation of her understanding of relationships and intimacy.', NULL, 12.99, 8.50, 0.00, 'Part I: The Journey, Part II: The Discovery, Part III: The Transformation', NULL, 3000000, TRUE),

('0156030423', '9780156030427', 'To the Lighthouse', NULL, 15, 15, 8, '1927-05-05', '1st Edition', 'English', 209, '8.0 x 5.2 x 0.6 inches', 170, 'A modernist novel following the Ramsay family and their guests during visits to their summer home on the Isle of Skye.', NULL, 'The novel is divided into three parts: "The Window," which takes place before World War I; "Time Passes," which covers the war years; and "The Lighthouse," which takes place after the war. It explores themes of time, memory, and the nature of human relationships.', NULL, 12.99, 8.00, 0.00, 'Part I: The Window, Part II: Time Passes, Part III: The Lighthouse', NULL, 300000, TRUE),

('1451673328', '9781451673326', 'The Martian Chronicles', NULL, 16, 16, 2, '1950-05-01', '1st Edition', 'English', 222, '8.0 x 5.2 x 0.6 inches', 180, 'A collection of interconnected short stories about the colonization of Mars and the eventual destruction of both Martian and human civilizations.', NULL, 'The book chronicles the colonization of Mars by humans fleeing a troubled Earth, and the eventual conflict between the Martians and the colonists. It explores themes of imperialism, environmental destruction, and the cyclical nature of history.', NULL, 13.99, 9.00, 0.00, 'January 1999: Rocket Summer through October 2026: The Million-Year Picnic', NULL, 800000, TRUE),

('0441013600', '9780441013609', 'The Dispossessed', NULL, 17, 17, 2, '1974-05-01', '1st Edition', 'English', 341, '8.0 x 5.2 x 0.9 inches', 280, 'A science fiction novel about a physicist who leaves his anarchist society to visit a capitalist world, exploring themes of politics, society, and human nature.', NULL, 'Shevek, a physicist from the anarchist moon Anarres, travels to the capitalist planet Urras to share his revolutionary theory. The novel alternates between his experiences on both worlds, exploring the strengths and weaknesses of different political systems.', NULL, 15.99, 10.50, 0.00, 'Part I: Anarres, Part II: Urras, Part III: Anarres, Part IV: Urras, Part V: Anarres, Part VI: Urras, Part VII: Anarres, Part VIII: Urras, Part IX: Anarres, Part X: Urras', NULL, 350000, TRUE),

('0099587132', '9780099587137', 'The Satanic Verses', NULL, 18, 18, 1, '1988-09-26', '1st Edition', 'English', 547, '8.0 x 5.2 x 1.2 inches', 450, 'A controversial novel about two Indian actors who survive a plane crash and experience supernatural transformations.', NULL, 'Gibreel Farishta and Saladin Chamcha, two Indian actors, survive a plane crash over the English Channel. They experience supernatural transformations that mirror the stories of the archangel Gabriel and the devil, exploring themes of identity, migration, and religious faith.', NULL, 16.99, 11.25, 0.00, 'Part I: The Angel Gibreel, Part II: Mahound, Part III: Ayesha, Part IV: A City Visible but Unseen, Part V: A Wedding, Part VI: Return to Jahilia, Part VII: The Angel Azraeel, Part VIII: The Parting of the Arabian Sea, Part IX: A Wonderful Lamp', NULL, 200000, TRUE),

('0395927212', '9780395927216', 'The Namesake', NULL, 19, 19, 1, '2003-09-01', '1st Edition', 'English', 291, '8.0 x 5.2 x 0.8 inches', 240, 'A novel about an Indian American family and their son''s struggle with his identity and the meaning of his name.', NULL, 'The novel follows the Ganguli family from their arrival in America through their son Gogol''s journey of self-discovery. Named after the Russian writer Nikolai Gogol, he struggles with his identity and eventually comes to understand the significance of his name and heritage.', NULL, 14.99, 9.75, 0.00, 'Part I: 1968, Part II: 1971, Part III: 1982, Part IV: 1994, Part V: 1999, Part VI: 2000, Part VII: 2001', NULL, 600000, TRUE),

('0385542372', '9780385542378', 'The Nickel Boys', NULL, 20, 20, 13, '2019-07-16', '1st Edition', 'English', 213, '8.2 x 5.5 x 0.7 inches', 180, 'A novel about two boys sent to a reform school in Florida during the Jim Crow era, based on a real institution.', NULL, 'Elwood Curtis, a black teenager in 1960s Florida, is sent to the Nickel Academy, a reform school that claims to provide vocational training but is actually a place of abuse and corruption. The novel follows his friendship with another boy and their struggle to survive the institution.', NULL, 15.99, 10.50, 0.00, 'Part I: Elwood, Part II: Turner, Part III: The Nickel Boys, Part IV: The Survivors', NULL, 700000, TRUE),

('0141036145', '9780141036145', 'Down and Out in Paris and London', NULL, 1, 1, 11, '1933-01-09', '1st Edition', 'English', 213, '7.8 x 5.1 x 0.6 inches', 180, 'A memoir about Orwell''s experiences living in poverty in Paris and London, exploring the lives of the working poor.', NULL, 'Orwell recounts his experiences working as a dishwasher in Paris and living among the homeless in London. The book provides a detailed account of poverty and working-class life in the early 20th century, written with Orwell''s characteristic clarity and social consciousness.', NULL, 11.99, 7.75, 0.00, 'Part I: Paris, Part II: London', NULL, 400000, TRUE),

('0141439855', '9780141439854', 'Sense and Sensibility', NULL, 2, 2, 8, '1811-10-30', '1st Edition', 'English', 409, '8.0 x 5.2 x 1.0 inches', 310, 'A novel about two sisters with contrasting approaches to life and love: one guided by sense, the other by sensibility.', NULL, 'Elinor and Marianne Dashwood are sisters with very different temperaments. Elinor is practical and restrained, while Marianne is emotional and romantic. Their experiences with love and loss teach them to balance sense and sensibility in their lives.', NULL, 10.99, 7.25, 5.00, 'Volume I: Chapters 1-22, Volume II: Chapters 1-14, Volume III: Chapters 1-14', NULL, 600000, TRUE),

('0060883294', '9780060883294', 'Chronicle of a Death Foretold', NULL, 3, 3, 1, '1981-01-01', '1st Edition', 'Spanish', 120, '8.0 x 5.3 x 0.4 inches', 100, 'A novella about the murder of Santiago Nasar, told through the recollections of various townspeople who knew the victim and killers.', NULL, 'The narrator returns to his hometown to investigate the murder of Santiago Nasar, which occurred twenty-seven years earlier. Through interviews with various townspeople, he reconstructs the events leading up to the murder and explores themes of honor, fate, and collective responsibility.', NULL, 12.99, 8.50, 0.00, 'Part I: The Investigation, Part II: The Recollections, Part III: The Murder, Part IV: The Aftermath', NULL, 800000, TRUE),

('0307743675', '9780307743671', 'The Stand', NULL, 4, 4, 2, '1978-09-01', '1st Edition', 'English', 1153, '8.2 x 5.5 x 1.9 inches', 950, 'A post-apocalyptic horror novel about the survivors of a plague that wipes out most of humanity, and their struggle between good and evil.', NULL, 'After a deadly plague kills 99% of the world''s population, the survivors are drawn to two locations: Boulder, Colorado, where they try to rebuild civilization, and Las Vegas, where a dark figure known as Randall Flagg is gathering his followers for a final confrontation.', NULL, 19.99, 13.25, 0.00, 'Part I: The Plague, Part II: The Journey, Part III: The Stand', NULL, 2000000, TRUE),

('0439708206', '9780439708203', 'Harry Potter and the Prisoner of Azkaban', NULL, 5, 5, 3, '1999-07-08', '1st Edition', 'English', 317, '7.6 x 5.0 x 1.2 inches', 280, 'The third novel in the Harry Potter series, following Harry''s third year at Hogwarts and the escape of Sirius Black from Azkaban.', NULL, 'Harry''s third year at Hogwarts is marked by the escape of Sirius Black, a dangerous prisoner from Azkaban who is believed to be after Harry. As Harry learns more about his parents'' past, he discovers that things are not always as they seem.', NULL, 13.99, 9.25, 0.00, 'Chapter 1: Owl Post through Chapter 22: Owl Post Again', NULL, 65000000, TRUE),

('006207350X', '9780062073501', 'The Murder of Roger Ackroyd', NULL, 6, 6, 4, '1926-06-01', '1st Edition', 'English', 288, '7.8 x 5.1 x 0.8 inches', 220, 'A detective novel featuring Hercule Poirot investigating the murder of a wealthy man in a small English village.', NULL, 'Roger Ackroyd is found murdered in his study, and Hercule Poirot is called in to investigate. The case becomes more complex when it''s revealed that Ackroyd was about to reveal a blackmailer''s identity, and several people had motives for his murder.', NULL, 13.99, 9.00, 5.00, 'Part I: The Facts, Part II: The Evidence, Part III: The Solution', NULL, 700000, TRUE),

('0684801485', '9780684801483', 'A Farewell to Arms', NULL, 7, 7, 1, '1929-09-27', '1st Edition', 'English', 332, '8.0 x 5.3 x 0.8 inches', 280, 'A novel set during World War I, following an American ambulance driver who falls in love with a British nurse.', NULL, 'Frederic Henry, an American serving as an ambulance driver in the Italian army during World War I, falls in love with Catherine Barkley, a British nurse. Their relationship develops against the backdrop of war, and they attempt to escape to Switzerland together.', NULL, 14.99, 9.75, 0.00, 'Book I: The Retreat, Book II: The Love, Book III: The Escape, Book IV: The Death, Book V: The End', NULL, 800000, TRUE),

('1400033437', '9781400033437', 'The Bluest Eye', NULL, 8, 8, 1, '1970-01-01', '1st Edition', 'English', 206, '8.0 x 5.2 x 0.6 inches', 170, 'A novel about a young black girl who longs for blue eyes, exploring themes of beauty, race, and self-worth.', NULL, 'Pecola Breedlove, a young black girl growing up in Ohio in the 1940s, believes that having blue eyes would make her beautiful and loved. The novel explores the devastating effects of internalized racism and the standards of beauty imposed by society.', NULL, 13.99, 9.00, 0.00, 'Autumn, Winter, Spring, Summer', NULL, 500000, TRUE),

('0375704043', '9780375704048', '1Q84', NULL, 9, 9, 1, '2009-05-29', '1st Edition', 'Japanese', 925, '8.0 x 5.2 x 1.8 inches', 800, 'A surreal novel set in an alternate version of 1984, following two characters whose lives become intertwined in mysterious ways.', NULL, 'Aomame, a fitness instructor and assassin, and Tengo, a math teacher and aspiring writer, find themselves in a world that seems to be 1984 but with subtle differences. Their stories gradually converge as they navigate this strange reality.', NULL, 22.99, 15.50, 0.00, 'Book 1: April-June, Book 2: July-September, Book 3: October-December', NULL, 1500000, TRUE),

('0385490835', '9780385490832', 'Oryx and Crake', NULL, 10, 10, 2, '2003-05-06', '1st Edition', 'English', 374, '8.0 x 5.2 x 1.0 inches', 300, 'A dystopian novel set in a post-apocalyptic world where genetic engineering has gone wrong, following the last human survivor.', NULL, 'Snowman, possibly the last human on Earth, lives among the Crakers, a genetically engineered species. Through flashbacks, he recounts the events that led to the destruction of human civilization and his role in the creation of the new world.', NULL, 16.99, 11.25, 0.00, 'Part I: Mangoes, Part II: SoYummie, Part III: MaddAddam', NULL, 600000, TRUE),

('0062255671', '9780062255679', 'Coraline', NULL, 11, 11, 3, '2002-07-02', '1st Edition', 'English', 162, '8.0 x 5.2 x 0.5 inches', 140, 'A dark fantasy novel about a young girl who discovers a parallel world that seems perfect but hides a sinister secret.', NULL, 'Coraline Jones discovers a door in her new home that leads to an alternate version of her life, where her "Other Mother" and "Other Father" seem perfect but have buttons for eyes. When she tries to return home, she finds herself trapped in this nightmarish world.', NULL, 12.99, 8.50, 0.00, 'Chapter 1: The Door, Chapter 2: The Other Mother, Chapter 3: The Other Father, Chapter 4: The Other World, Chapter 5: The Escape, Chapter 6: The Return, Chapter 7: The Rescue, Chapter 8: The End', NULL, 2500000, TRUE),

('0307277738', '9780307277736', 'Americanah', NULL, 12, 12, 1, '2013-05-14', '1st Edition', 'English', 477, '8.0 x 5.2 x 1.2 inches', 400, 'A novel about a young Nigerian woman who immigrates to America and later returns to Nigeria, exploring themes of race, identity, and belonging.', NULL, 'Ifemelu leaves Nigeria for America to attend university, while her boyfriend Obinze stays behind. The novel follows their separate journeys and eventual reunion, exploring themes of immigration, race relations in America, and the experience of returning home.', NULL, 16.99, 11.25, 0.00, 'Part I: Princeton, Part II: Lagos, Part III: London, Part IV: Lagos', NULL, 800000, TRUE),

('0307277690', '9780307277699', 'An Artist of the Floating World', NULL, 13, 13, 1, '1986-05-01', '1st Edition', 'English', 206, '8.0 x 5.2 x 0.6 inches', 170, 'A novel about an aging Japanese artist reflecting on his life and career during the post-World War II period.', NULL, 'Masuji Ono, a retired artist, reflects on his life and career during the post-war period in Japan. As he arranges his daughter''s marriage, he must confront his past involvement with the nationalist movement and the consequences of his actions.', NULL, 12.99, 8.50, 0.00, 'Part I: October 1948, Part II: April 1949, Part III: November 1949, Part IV: June 1950', NULL, 400000, TRUE),

('0061122432', '9780061122439', 'The Zahir', NULL, 14, 14, 1, '2005-06-01', '1st Edition', 'Portuguese', 336, '7.8 x 5.1 x 0.8 inches', 280, 'A novel about a successful writer whose wife disappears, leading him on a journey of self-discovery and spiritual awakening.', NULL, 'A successful writer''s wife Esther disappears without explanation, leaving behind only a note. As he searches for her, he encounters a young man who claims to know where she is, leading him on a journey that challenges his understanding of love, freedom, and the meaning of life.', NULL, 13.99, 9.25, 0.00, 'Part I: The Disappearance, Part II: The Search, Part III: The Discovery, Part IV: The Transformation', NULL, 2000000, TRUE),

('0156030431', '9780156030434', 'Orlando', NULL, 15, 15, 8, '1928-10-11', '1st Edition', 'English', 228, '8.0 x 5.2 x 0.7 inches', 190, 'A satirical novel about a nobleman who lives for centuries and changes gender, exploring themes of time, gender, and literature.', NULL, 'Orlando begins as a young nobleman in Elizabethan England and lives for over 300 years, eventually becoming a woman. The novel explores themes of gender, time, and the nature of identity through Orlando''s fantastical journey through history.', NULL, 13.99, 9.00, 0.00, 'Chapter 1: The Beginning, Chapter 2: The Middle, Chapter 3: The End', NULL, 250000, TRUE),

('1451673336', '9781451673333', 'Something Wicked This Way Comes', NULL, 16, 16, 7, '1962-09-17', '1st Edition', 'English', 293, '8.0 x 5.2 x 0.8 inches', 240, 'A dark fantasy novel about two boys who encounter a sinister traveling carnival that preys on people''s deepest desires.', NULL, 'Will Halloway and Jim Nightshade are best friends who encounter a mysterious carnival that arrives in their town. The carnival''s owner, Mr. Dark, offers to fulfill people''s deepest wishes, but at a terrible cost. The boys must resist the carnival''s temptations and save their town.', NULL, 13.99, 9.00, 0.00, 'Part I: The Arrival, Part II: The Temptation, Part III: The Confrontation, Part IV: The Victory', NULL, 600000, TRUE),

('0441013618', '9780441013616', 'The Lathe of Heaven', NULL, 17, 17, 2, '1971-03-01', '1st Edition', 'English', 184, '8.0 x 5.2 x 0.5 inches', 150, 'A science fiction novel about a man whose dreams can change reality, and the psychiatrist who tries to use this power for good.', NULL, 'George Orr discovers that his dreams can alter reality, but the changes are often unintended and catastrophic. His psychiatrist, Dr. Haber, tries to use this power to solve world problems, but each attempt leads to new and worse problems.', NULL, 12.99, 8.50, 0.00, 'Part I: The Discovery, Part II: The Experiments, Part III: The Consequences, Part IV: The Resolution', NULL, 300000, TRUE),

('0099587140', '9780099587144', 'Shame', NULL, 18, 18, 1, '1983-09-01', '1st Edition', 'English', 304, '8.0 x 5.2 x 0.8 inches', 250, 'A novel set in a fictional country that resembles Pakistan, exploring themes of politics, power, and the nature of shame.', NULL, 'The novel follows the lives of Omar Khayyam Shakil and his three mothers in a fictional country that mirrors Pakistan''s history. It explores themes of political corruption, family dynamics, and the destructive power of shame and humiliation.', NULL, 14.99, 9.75, 0.00, 'Part I: The Family, Part II: The Politics, Part III: The Fall, Part IV: The Aftermath', NULL, 300000, TRUE),

('0395927220', '9780395927223', 'Unaccustomed Earth', NULL, 19, 19, 1, '2008-04-01', '1st Edition', 'English', 333, '8.0 x 5.2 x 0.9 inches', 280, 'A collection of short stories exploring the lives of Bengali Americans and their experiences with family, love, and cultural identity.', NULL, 'This collection of eight short stories explores the lives of Bengali Americans, focusing on themes of family relationships, cultural displacement, and the immigrant experience. The stories range from tales of young love to complex family dynamics.', NULL, 15.99, 10.50, 0.00, 'Part I: Unaccustomed Earth, Part II: Hell-Heaven, Part III: A Choice of Accommodations, Part IV: Only Goodness, Part V: Nobody''s Business, Part VI: Once in a Lifetime, Part VII: Year''s End, Part VIII: Going Ashore', NULL, 400000, TRUE),

('0385542380', '9780385542385', 'Zone One', NULL, 20, 20, 2, '2011-10-18', '1st Edition', 'English', 259, '8.2 x 5.5 x 0.8 inches', 220, 'A post-apocalyptic novel about a man working to clear Manhattan of zombies after a plague has devastated the world.', NULL, 'Mark Spitz is a "sweeper" working to clear Manhattan of the remaining zombies after a plague has devastated the world. As he works in Zone One, the last area to be cleared, he reflects on his past and the nature of survival in a post-apocalyptic world.', NULL, 14.99, 9.75, 0.00, 'Part I: Friday, Part II: Saturday, Part III: Sunday', NULL, 350000, TRUE),

('0141036146', '9780141036146', 'Homage to Catalonia', NULL, 1, 1, 11, '1938-04-25', '1st Edition', 'English', 232, '7.8 x 5.1 x 0.6 inches', 200, 'A memoir about Orwell''s experiences fighting in the Spanish Civil War, providing a firsthand account of the conflict.', NULL, 'Orwell recounts his experiences fighting for the Republican side in the Spanish Civil War, including his time in the trenches and his disillusionment with the political infighting among the Republican factions. The book provides a detailed account of the war and its political complexities.', NULL, 12.99, 8.50, 0.00, 'Part I: The War, Part II: The Politics, Part III: The Retreat', NULL, 300000, TRUE),

('0141439863', '9780141439861', 'Mansfield Park', NULL, 2, 2, 8, '1814-07-01', '1st Edition', 'English', 507, '8.0 x 5.2 x 1.2 inches', 380, 'A novel about a young woman who goes to live with her wealthy relatives and becomes involved in their family drama.', NULL, 'Fanny Price, a poor relation, goes to live with her wealthy aunt and uncle at Mansfield Park. She observes the family''s moral failings and eventually becomes the moral center of the household, teaching them about true virtue and happiness.', NULL, 11.99, 7.75, 5.00, 'Volume I: Chapters 1-18, Volume II: Chapters 1-13, Volume III: Chapters 1-17', NULL, 500000, TRUE),

('0060883308', '9780060883300', 'The General in His Labyrinth', NULL, 3, 3, 13, '1989-01-01', '1st Edition', 'Spanish', 285, '8.0 x 5.3 x 0.8 inches', 240, 'A historical novel about the final days of Simón Bolívar, the liberator of South America.', NULL, 'The novel follows Simón Bolívar during his final journey down the Magdalena River in 1830, as he reflects on his life, his achievements, and his failures. It presents a complex portrait of a man who liberated much of South America but died disillusioned and alone.', NULL, 13.99, 9.25, 0.00, 'Part I: The Journey Begins, Part II: The Memories, Part III: The End', NULL, 400000, TRUE),

('0307743683', '9780307743688', 'Carrie', NULL, 4, 4, 7, '1974-04-05', '1st Edition', 'English', 199, '8.2 x 5.5 x 0.6 inches', 160, 'A horror novel about a teenage girl with telekinetic powers who is bullied by her classmates and seeks revenge.', NULL, 'Carrie White is a shy, awkward teenager who discovers she has telekinetic powers. After being humiliated at her senior prom, she uses her powers to exact revenge on her tormentors, leading to a catastrophic climax.', NULL, 12.99, 8.50, 0.00, 'Part I: Blood Sport, Part II: Prom Night, Part III: Wreckage', NULL, 1000000, TRUE),

('0439708214', '9780439708210', 'Harry Potter and the Goblet of Fire', NULL, 5, 5, 3, '2000-07-08', '1st Edition', 'English', 636, '7.6 x 5.0 x 1.8 inches', 560, 'The fourth novel in the Harry Potter series, following Harry''s participation in the Triwizard Tournament and the return of Voldemort.', NULL, 'Harry''s fourth year at Hogwarts is marked by the Triwizard Tournament, a dangerous competition between three wizarding schools. When Harry is mysteriously entered into the tournament, he must compete against older, more experienced students while dealing with the growing threat of Voldemort''s return.', NULL, 15.99, 10.75, 0.00, 'Chapter 1: The Riddle House through Chapter 37: The Beginning', NULL, 55000000, TRUE),

('0062073518', '9780062073518', 'And Then There Were None', NULL, 6, 6, 4, '1939-11-06', '1st Edition', 'English', 264, '7.8 x 5.1 x 0.7 inches', 200, 'A mystery novel about ten people who are invited to an island and then killed one by one according to a nursery rhyme.', NULL, 'Ten strangers are invited to a remote island by a mysterious host. When they arrive, they discover that their host is absent, and they are killed one by one according to the pattern of a nursery rhyme. The survivors must figure out who the killer is before they all die.', NULL, 12.99, 8.50, 5.00, 'Part I: The Arrival, Part II: The Murders, Part III: The Solution', NULL, 100000000, TRUE),

('0684801493', '9780684801490', 'The Sun Also Rises', NULL, 7, 7, 1, '1926-10-22', '1st Edition', 'English', 251, '8.0 x 5.3 x 0.7 inches', 210, 'A novel about a group of expatriates in Paris and their trip to Spain for the running of the bulls, exploring themes of love, masculinity, and the lost generation.', NULL, 'Jake Barnes, a World War I veteran, lives in Paris with other expatriates. He is in love with Lady Brett Ashley, but their relationship is complicated by his war injury. The group travels to Spain for the running of the bulls, where their relationships and tensions come to a head.', NULL, 13.99, 9.25, 0.00, 'Book I: Paris, Book II: Spain, Book III: The Return', NULL, 700000, TRUE),

('1400033445', '9781400033444', 'Jazz', NULL, 8, 8, 1, '1992-04-01', '1st Edition', 'English', 229, '8.0 x 5.2 x 0.7 inches', 190, 'A novel set in 1920s Harlem, exploring themes of love, violence, and the African American experience during the Jazz Age.', NULL, 'The novel follows the lives of Joe and Violet Trace, a married couple living in Harlem during the 1920s. When Joe has an affair with a young woman named Dorcas, the resulting violence and its aftermath explore themes of love, betrayal, and the search for identity.', NULL, 13.99, 9.00, 0.00, 'Part I: The City, Part II: The Country, Part III: The City', NULL, 300000, TRUE),

('0375704051', '9780375704055', 'Hard-Boiled Wonderland and the End of the World', NULL, 9, 9, 2, '1985-06-15', '1st Edition', 'Japanese', 400, '8.0 x 5.2 x 1.0 inches', 340, 'A surreal novel that alternates between two parallel stories: a cyberpunk tale and a fantasy story about a man in a walled town.', NULL, 'The novel alternates between two stories: one about a "Calcutec" who processes data in a cyberpunk Tokyo, and another about a man who arrives in a mysterious walled town where he must read dreams from unicorn skulls. The stories gradually converge in unexpected ways.', NULL, 15.99, 10.50, 0.00, 'Chapters 1-40 alternating between the two parallel stories', NULL, 800000, TRUE),

('0385490843', '9780385490849', 'The Year of the Flood', NULL, 10, 10, 2, '2009-09-22', '1st Edition', 'English', 431, '8.0 x 5.2 x 1.1 inches', 350, 'A dystopian novel set in the same world as Oryx and Crake, following different characters during the same apocalyptic events.', NULL, 'The novel follows Toby and Ren, two women who survive the same plague that destroyed human civilization in Oryx and Crake. Their stories intersect with the events of the previous novel, providing a different perspective on the apocalypse and its aftermath.', NULL, 16.99, 11.25, 0.00, 'Part I: The Flood, Part II: The Garden, Part III: The Year of the Flood', NULL, 500000, TRUE),

('0062255689', '9780062255686', 'The Graveyard Book', NULL, 11, 11, 3, '2008-09-30', '1st Edition', 'English', 312, '8.0 x 5.2 x 0.8 inches', 260, 'A fantasy novel about a boy who is raised by ghosts in a graveyard after his family is murdered.', NULL, 'Nobody Owens, known as Bod, is raised by ghosts in a graveyard after his family is murdered by a man named Jack. As he grows up among the dead, he learns their secrets and must eventually face the man who killed his family.', NULL, 13.99, 9.25, 0.00, 'Chapter 1: How Nobody Came to the Graveyard through Chapter 8: Leavings and Partings', NULL, 3000000, TRUE),

('0307277746', '9780307277743', 'We Should All Be Feminists', NULL, 12, 12, 11, '2014-07-29', '1st Edition', 'English', 52, '8.0 x 5.2 x 0.2 inches', 50, 'An essay adapted from Adichie''s TEDx talk about feminism and gender equality in the 21st century.', NULL, 'Based on her famous TEDx talk, Adichie explores what it means to be a feminist in the 21st century. She discusses the ways in which gender inequality affects both men and women and calls for a more inclusive and intersectional approach to feminism.', NULL, 8.99, 5.75, 0.00, 'Introduction, The Problem, The Solution, Conclusion', NULL, 2000000, TRUE),

('0307277704', '9780307277705', 'The Buried Giant', NULL, 13, 13, 3, '2015-03-03', '1st Edition', 'English', 317, '8.0 x 5.2 x 0.8 inches', 260, 'A fantasy novel set in post-Arthurian Britain, following an elderly couple on a journey to find their son.', NULL, 'Axl and Beatrice, an elderly couple living in a Britain where people have lost their memories due to a mist, set out on a journey to find their son. Along the way, they encounter knights, dragons, and other fantastical elements while trying to recover their lost memories.', NULL, 14.99, 9.75, 0.00, 'Part I: The Journey Begins, Part II: The Encounters, Part III: The Discovery, Part IV: The End', NULL, 600000, TRUE),

('0061122440', '9780061122446', 'Brida', NULL, 14, 14, 1, '1990-01-01', '1st Edition', 'Portuguese', 256, '7.8 x 5.1 x 0.7 inches', 210, 'A novel about a young Irish woman''s spiritual journey and her search for love and meaning in life.', NULL, 'Brida, a young Irish woman, seeks to learn about magic and spirituality. She meets a wise man who teaches her about the traditions of the sun and a woman who teaches her about the traditions of the moon, leading her on a journey of self-discovery and spiritual awakening.', NULL, 12.99, 8.50, 0.00, 'Part I: The Search, Part II: The Learning, Part III: The Choice, Part IV: The Love', NULL, 1500000, TRUE),

('0156030449', '9780156030441', 'The Waves', NULL, 15, 15, 8, '1931-10-08', '1st Edition', 'English', 297, '8.0 x 5.2 x 0.8 inches', 250, 'A modernist novel following six friends from childhood to middle age, told through their interior monologues.', NULL, 'The novel follows six friends—Bernard, Susan, Rhoda, Neville, Jinny, and Louis—from childhood to middle age. Their stories are told through interior monologues that explore themes of identity, time, and the nature of human consciousness.', NULL, 13.99, 9.00, 0.00, 'Part I: Childhood, Part II: School, Part III: University, Part IV: Adulthood, Part V: Middle Age, Part VI: Old Age', NULL, 200000, TRUE),

('1451673344', '9781451673340', 'Dandelion Wine', NULL, 16, 16, 1, '1957-09-01', '1st Edition', 'English', 239, '8.0 x 5.2 x 0.7 inches', 200, 'A semi-autobiographical novel about a young boy''s magical summer in a small Illinois town in 1928.', NULL, 'Douglas Spaulding, a 12-year-old boy, experiences a magical summer in Green Town, Illinois, in 1928. The novel captures the wonder and nostalgia of childhood summers, with each chapter focusing on different experiences and discoveries.', NULL, 12.99, 8.50, 0.00, 'Part I: The Beginning, Part II: The Adventures, Part III: The End', NULL, 400000, TRUE),

('0441013626', '9780441013623', 'The Left Hand of Darkness', NULL, 17, 17, 2, '1969-03-01', '1st Edition', 'English', 304, '8.0 x 5.2 x 0.8 inches', 250, 'A science fiction novel about a human envoy on a planet where inhabitants can change gender, exploring themes of gender and society.', NULL, 'Genly Ai, an envoy from the Ekumen, travels to the planet Gethen to convince its inhabitants to join an interplanetary alliance. On this world where people are neither male nor female but can become either during their monthly cycle, Ai must navigate complex political and cultural differences.', NULL, 15.99, 10.50, 0.00, 'Chapters 1-20 following Genly Ai''s mission and experiences', NULL, 400000, TRUE),

('0099587158', '9780099587151', 'The Moor''s Last Sigh', NULL, 18, 18, 1, '1995-09-01', '1st Edition', 'English', 435, '8.0 x 5.2 x 1.1 inches', 360, 'A novel about the last surviving member of a wealthy Indian family, exploring themes of history, identity, and the end of an era.', NULL, 'Moraes Zogoiby, the last surviving member of the wealthy da Gama-Zogoiby family, tells the story of his family''s rise and fall. The novel spans several generations and explores themes of history, identity, and the changing face of India.', NULL, 15.99, 10.50, 0.00, 'Part I: The Beginning, Part II: The Rise, Part III: The Fall, Part IV: The End', NULL, 250000, TRUE),

('0395927238', '9780395927240', 'The Lowland', NULL, 19, 19, 1, '2013-09-24', '1st Edition', 'English', 339, '8.0 x 5.2 x 0.9 inches', 280, 'A novel about two brothers in 1960s India, one who becomes involved in the Naxalite movement and the other who moves to America.', NULL, 'Subhash and Udayan are brothers growing up in 1960s Calcutta. When Udayan becomes involved in the Naxalite movement and is killed, Subhash moves to America and marries Udayan''s widow, Gauri. The novel explores themes of family, politics, and the immigrant experience.', NULL, 15.99, 10.50, 0.00, 'Part I: The Brothers, Part II: The Marriage, Part III: The Family, Part IV: The Return', NULL, 500000, TRUE),

('0385542398', '9780385542392', 'The Intuitionist', NULL, 20, 20, 1, '1999-01-01', '1st Edition', 'English', 255, '8.2 x 5.5 x 0.7 inches', 210, 'A novel about an elevator inspector who becomes embroiled in a conspiracy involving elevator accidents and racial politics.', NULL, 'Lila Mae Watson is the first black female elevator inspector in a city where elevators are a matter of life and death. When an elevator she inspected crashes, she becomes embroiled in a conspiracy that involves racial politics, corporate corruption, and the search for the perfect elevator.', NULL, 13.99, 9.25, 0.00, 'Part I: The Crash, Part II: The Investigation, Part III: The Conspiracy, Part IV: The Resolution', NULL, 200000, TRUE);


