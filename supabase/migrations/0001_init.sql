-- CRAG Study Agent — initial schema
-- Run this in the Supabase SQL editor (or via `supabase db push`).

create extension if not exists vector;

-- ---------- documents ----------
create table if not exists documents (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    name text not null,
    subject text,
    file_path text not null,
    created_at timestamptz not null default now()
);

-- ---------- document_chunks ----------
create table if not exists document_chunks (
    id uuid primary key default gen_random_uuid(),
    document_id uuid not null references documents(id) on delete cascade,
    content text not null,
    embedding vector(384), -- sentence-transformers/all-MiniLM-L6-v2 dimension
    page_number int,
    chunk_index int,
    metadata jsonb default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists document_chunks_embedding_idx
    on document_chunks using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

create index if not exists document_chunks_document_id_idx
    on document_chunks (document_id);

-- ---------- conversations ----------
create table if not exists conversations (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade,
    title text,
    created_at timestamptz not null default now()
);

-- ---------- messages ----------
create table if not exists messages (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid not null references conversations(id) on delete cascade,
    role text not null check (role in ('user', 'assistant')),
    content text not null,
    created_at timestamptz not null default now()
);

-- ---------- retrieval_logs ----------
create table if not exists retrieval_logs (
    id uuid primary key default gen_random_uuid(),
    query text not null,
    retrieved_chunks uuid[],
    confidence_score float,
    correction_attempts int default 0,
    final_status text,
    response_time int, -- milliseconds
    created_at timestamptz not null default now()
);

-- ---------- vector similarity search RPC ----------
-- Called from app/services/retrieval_service.py::vector_search
create or replace function match_document_chunks (
    query_embedding vector(384),
    match_count int default 10,
    filter_document_ids uuid[] default null
)
returns table (
    id uuid,
    document_id uuid,
    content text,
    page_number int,
    chunk_index int,
    metadata jsonb,
    similarity float
)
language sql stable
as $$
    select
        document_chunks.id,
        document_chunks.document_id,
        document_chunks.content,
        document_chunks.page_number,
        document_chunks.chunk_index,
        document_chunks.metadata,
        1 - (document_chunks.embedding <=> query_embedding) as similarity
    from document_chunks
    where filter_document_ids is null
       or document_chunks.document_id = any(filter_document_ids)
    order by document_chunks.embedding <=> query_embedding
    limit match_count;
$$;

-- ---------- row level security ----------
alter table documents enable row level security;
alter table conversations enable row level security;
alter table messages enable row level security;

create policy "Users manage their own documents"
    on documents for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

create policy "Users manage their own conversations"
    on conversations for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

create policy "Users manage messages in their conversations"
    on messages for all
    using (
        exists (
            select 1 from conversations
            where conversations.id = messages.conversation_id
            and conversations.user_id = auth.uid()
        )
    );

-- document_chunks and retrieval_logs are accessed only via the backend's
-- service-role key, so RLS is intentionally left off for them.
