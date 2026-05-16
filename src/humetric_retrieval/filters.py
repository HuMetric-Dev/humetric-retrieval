from __future__ import annotations

import psycopg
from humetric_core import Err, Ok, ParsedQuery, Result, normalize_skill
from psycopg import sql

from humetric_retrieval.errors import FilterFailed, RetrievalError


def candidate_ids(
    conn: psycopg.Connection, parsed: ParsedQuery, limit: int = 5000
) -> Result[set[str] | None, RetrievalError]:
    """Apply hard filters from ParsedQuery and return the allowed person id set.

    Returns ``Ok(None)`` when the query specifies no filter constraints — the
    caller treats that as "filter disabled, every candidate is allowed."
    Returns ``Ok(set)`` when at least one constraint was specified; the set
    may be empty if the constraints together yield zero matches, in which
    case the caller should honestly return zero candidates rather than
    silently fall through to "show everything."
    """
    has_constraint = (
        parsed.location is not None or parsed.min_followers is not None or bool(parsed.must_skills)
    )
    if not has_constraint:
        return Ok[set[str] | None](None)

    conditions: list[sql.Composable] = []
    params: list[object] = []

    if parsed.location:
        conditions.append(sql.SQL("LOWER(p.location) LIKE %s"))
        params.append(f"%{parsed.location.strip().lower()}%")

    if parsed.min_followers is not None:
        conditions.append(sql.SQL("p.follower_count >= %s"))
        params.append(int(parsed.min_followers))

    must_normalized = [normalize_skill(s) for s in parsed.must_skills]

    try:
        with conn.cursor() as cur:
            if not must_normalized:
                stmt: sql.Composable = sql.SQL("SELECT p.id FROM persons p")
                if conditions:
                    stmt = stmt + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)
                stmt = stmt + sql.SQL(" LIMIT %s")
                params.append(limit)
                cur.execute(stmt, params)
            else:
                placeholders = sql.SQL(", ").join([sql.Placeholder()] * len(must_normalized))
                stmt = (
                    sql.SQL(
                        "SELECT p.id FROM persons p "
                        "JOIN person_skills ps ON ps.person_id = p.id "
                        "JOIN skills s ON s.name = ps.skill_name "
                        "WHERE s.normalized IN ("
                    )
                    + placeholders
                    + sql.SQL(")")
                )
                params_with_skills: list[object] = [*must_normalized]
                if conditions:
                    stmt = stmt + sql.SQL(" AND ") + sql.SQL(" AND ").join(conditions)
                    params_with_skills.extend(params[: len(conditions)])
                stmt = stmt + sql.SQL(
                    " GROUP BY p.id HAVING COUNT(DISTINCT s.normalized) = %s LIMIT %s"
                )
                params_with_skills.append(len(must_normalized))
                params_with_skills.append(limit)
                cur.execute(stmt, params_with_skills)
            rows = cur.fetchall()
    except psycopg.Error as e:
        return Err(FilterFailed(detail=str(e)))

    return Ok[set[str] | None]({row[0] for row in rows})
