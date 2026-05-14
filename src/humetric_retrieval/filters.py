from __future__ import annotations

import sqlite3

from humetric_core import Err, Ok, ParsedQuery, Result, normalize_skill

from humetric_retrieval.errors import FilterFailed, RetrievalError


def candidate_ids(
    conn: sqlite3.Connection, parsed: ParsedQuery, limit: int = 5000
) -> Result[set[str], RetrievalError]:
    """Apply hard filters from ParsedQuery and return the allowed person id set.

    No `must_skills` → no skill join → faster. Empty result is *not* an
    error; downstream branches will simply intersect against an empty set,
    which the caller can detect.
    """
    conditions: list[str] = []
    params: list[object] = []

    if parsed.location:
        conditions.append("LOWER(p.location) LIKE ?")
        params.append(f"%{parsed.location.strip().lower()}%")

    if parsed.min_followers is not None:
        conditions.append("p.follower_count >= ?")
        params.append(int(parsed.min_followers))

    must_normalized = [normalize_skill(s) for s in parsed.must_skills]

    try:
        if not must_normalized:
            sql = "SELECT p.id FROM persons p"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        else:
            # All must-skills must appear in this person's skills.
            sql = """
                SELECT p.id FROM persons p
                JOIN person_skills ps ON ps.person_id = p.id
                JOIN skills s ON s.name = ps.skill_name
                WHERE s.normalized IN ({placeholders})
            """.format(placeholders=",".join("?" * len(must_normalized)))
            params_with_skills: list[object] = [*must_normalized]
            if conditions:
                sql += " AND " + " AND ".join(conditions)
                params_with_skills.extend(params[: len(conditions)])
            sql += " GROUP BY p.id HAVING COUNT(DISTINCT s.normalized) = ? LIMIT ?"
            params_with_skills.append(len(must_normalized))
            params_with_skills.append(limit)
            rows = conn.execute(sql, params_with_skills).fetchall()
    except sqlite3.Error as e:
        return Err(FilterFailed(detail=str(e)))

    return Ok({row["id"] for row in rows})
