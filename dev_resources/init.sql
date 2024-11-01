CREATE TABLE preference (
    id uuid NOT NULL PRIMARY KEY,
    prompt text,
    preferred text,
    rejected text
)