-- Preserve existing data. Legacy documents must be re-uploaded for trustworthy OCR/model metadata.
alter table public.documents add column if not exists status text not null default 'needs_reindex';
alter table public.documents add column if not exists page_count integer not null default 0;
alter table public.documents add column if not exists processed_pages integer not null default 0;
alter table public.documents add column if not exists chunk_count integer not null default 0;
alter table public.documents add column if not exists ingestion jsonb not null default '{}'::jsonb;
alter table public.messages add column if not exists metadata jsonb not null default '{}'::jsonb;
alter table public.retrieval_logs add column if not exists user_id uuid references auth.users(id) on delete cascade;
alter table public.retrieval_logs add column if not exists conversation_id uuid references public.conversations(id) on delete set null;
alter table public.retrieval_logs add column if not exists metadata jsonb not null default '{}'::jsonb;

create index if not exists study_documents_owner_idx on public.documents(user_id, created_at);
create index if not exists study_conversations_owner_idx on public.conversations(user_id, created_at);
create index if not exists study_messages_conversation_idx on public.messages(conversation_id, created_at);
create index if not exists study_logs_owner_idx on public.retrieval_logs(user_id, created_at);
create index if not exists study_chunks_order_idx on public.document_chunks(document_id, chunk_index);

-- The UI uses Supabase only for Auth. Table and RPC access is through verified backend routes.
alter table public.document_chunks enable row level security;
alter table public.retrieval_logs enable row level security;
alter table public.documents enable row level security;
alter table public.conversations enable row level security;
alter table public.messages enable row level security;
revoke all on public.documents, public.document_chunks, public.conversations, public.messages, public.retrieval_logs from anon, authenticated;
grant all on public.documents, public.document_chunks, public.conversations, public.messages, public.retrieval_logs to service_role;

create or replace function public.match_study_chunks(
    query_embedding vector(384), match_count integer,
    filter_document_ids uuid[], query_model text, query_version text
) returns table (
    id uuid, document_id uuid, content text, page_number integer, chunk_index integer,
    metadata jsonb, similarity double precision
) language sql stable security invoker set search_path = public, extensions as $$
    -- Exact search over the authorized document subset avoids ANN post-filter recall loss.
    with eligible as materialized (
        select c.* from public.document_chunks c join public.documents d on d.id = c.document_id
        where c.document_id = any(filter_document_ids)
          and d.status in ('ready', 'partial')
          and c.metadata->>'embedding_model' = query_model
          and c.metadata->>'embedding_version' = query_version
          and c.embedding is not null
    )
    select e.id, e.document_id, e.content, e.page_number, e.chunk_index, e.metadata,
           (1 - (e.embedding <=> query_embedding))::double precision
    from eligible e order by e.embedding <=> query_embedding
    limit least(greatest(match_count, 1), 200);
$$;
revoke execute on function public.match_study_chunks(vector, integer, uuid[], text, text) from public, anon, authenticated;
grant execute on function public.match_study_chunks(vector, integer, uuid[], text, text) to service_role;

-- Close legacy similarity RPCs too; signatures can differ in existing installations.
do $$
declare fn regprocedure;
begin
    for fn in select p.oid::regprocedure from pg_proc p join pg_namespace n on n.oid=p.pronamespace
              where n.nspname='public' and p.proname='match_document_chunks'
    loop
        execute format('revoke execute on function %s from public, anon, authenticated', fn);
        execute format('grant execute on function %s to service_role', fn);
    end loop;
end $$;

create or replace function public.study_analytics(owner_id uuid)
returns jsonb language sql stable security invoker set search_path=public as $$
    select jsonb_build_object(
        'total_documents', (select count(*) from public.documents where user_id=owner_id),
        'total_questions', count(*),
        'average_confidence', coalesce(avg(confidence_score),0),
        'average_correction_attempts', coalesce(avg(correction_attempts),0),
        'average_response_time_ms', coalesce(avg(response_time),0),
        'supported_answers', count(*) filter(where final_status='supported'),
        'validation_unavailable', count(*) filter(where final_status='unavailable'),
        'fallback_usage_rate', 0
    ) from public.retrieval_logs where user_id=owner_id;
$$;
revoke execute on function public.study_analytics(uuid) from public, anon, authenticated;
grant execute on function public.study_analytics(uuid) to service_role;

insert into storage.buckets(id, name, public, file_size_limit, allowed_mime_types)
values ('study-originals', 'study-originals', false, 31457280,
        array['application/pdf','application/json','text/plain','image/jpeg','image/png','image/webp'])
on conflict (id) do update set public=false;

notify pgrst, 'reload schema';
