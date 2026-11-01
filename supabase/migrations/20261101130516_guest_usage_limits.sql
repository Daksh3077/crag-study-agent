-- Allowances start when this feature is deployed; existing content is preserved.
-- Counters belong to the guest identity, not conversations or document rows.
create table public.guest_usage (
    user_id uuid primary key references auth.users(id) on delete cascade,
    questions integer not null default 0 check (questions between 0 and 5),
    uploads integer not null default 0 check (uploads between 0 and 2)
);
alter table public.guest_usage enable row level security;
revoke all on public.guest_usage from public, anon, authenticated;
grant select, insert, update on public.guest_usage to service_role;

create function public.consume_guest_usage(p_user_id uuid, p_kind text)
returns boolean
language plpgsql
security invoker
set search_path = ''
as $$
declare affected integer;
begin
    if p_kind is null or p_kind not in ('questions', 'uploads') then
        raise exception 'Unknown usage kind';
    end if;
    insert into public.guest_usage(user_id) values (p_user_id)
        on conflict (user_id) do nothing;
    -- Conditional UPDATE takes a row lock and rechecks after concurrent updates.
    if p_kind = 'questions' then
        update public.guest_usage set questions = questions + 1
            where user_id = p_user_id and questions < 5;
    else
        update public.guest_usage set uploads = uploads + 1
            where user_id = p_user_id and uploads < 2;
    end if;
    get diagnostics affected = row_count;
    return affected = 1;
end;
$$;
revoke all on function public.consume_guest_usage(uuid, text) from public, anon, authenticated;
grant execute on function public.consume_guest_usage(uuid, text) to service_role;
