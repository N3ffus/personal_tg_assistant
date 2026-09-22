"""Cypher the Graphiti adapter runs directly, beside Graphiti's own queries.

Every read and erase matches the namespace, so a ref from another user reaches
nothing.
"""

# A date never becomes an entity, so an edge like HAS_DATE_OF_BIRTH loses its
# target and is dropped: "родился 17.11.2001" can only survive in the episode.
# Episode search is BM25 — "дата рождения" does not match "родился" — so the
# adapter embeds every episode itself and recalls it by meaning.
EPISODE_SIMILARITY = """
MATCH (episode:Episodic {group_id: $namespace})
WHERE episode.content_embedding IS NOT NULL
WITH episode,
     vector.similarity.cosine(episode.content_embedding, $embedding) AS score
WHERE score >= $min_score
RETURN episode.uuid AS uuid, episode.name AS name, episode.content AS content,
       episode.valid_at AS valid_at
ORDER BY score DESC LIMIT $limit
"""

READ_EDGE_FACTS = """
MATCH (:Entity {group_id: $namespace})-[fact:RELATES_TO]->(:Entity)
WHERE fact.group_id = $namespace AND fact.uuid IN $uuids
RETURN fact.fact AS text
"""
READ_EPISODE_CONTENTS = """
MATCH (episode:Episodic {group_id: $namespace})
WHERE episode.uuid IN $uuids
RETURN episode.content AS content
"""
FORGET_EDGES = """
MATCH (source:Entity {group_id: $namespace})-[fact:RELATES_TO]->(target:Entity)
WHERE fact.group_id = $namespace AND (fact.uuid IN $uuids OR fact.fact IN $texts)
WITH fact, fact.fact AS text, source.uuid AS source, target.uuid AS target
DELETE fact
RETURN text, source, target
"""
FORGET_EPISODES = """
MATCH (episode:Episodic {group_id: $namespace})
WHERE episode.uuid IN $uuids OR episode.content IN $contents
WITH episode, episode.content AS content
DETACH DELETE episode
RETURN content
"""
READ_SUMMARIES = """
MATCH (node:Entity {group_id: $namespace})
WHERE node.summary IS NOT NULL AND node.summary <> ''
RETURN node.uuid AS uuid, node.summary AS summary
"""
SET_SUMMARY = """
MATCH (node:Entity {group_id: $namespace, uuid: $uuid})
SET node.summary = $summary
"""
# An entity left with nothing to say (no summary, fact or episode naming it)
# would still surface its bare name, "Ozon", and invite a guess.
DROP_ORPHANS = """
MATCH (node:Entity {group_id: $namespace})
WHERE node.uuid IN $uuids AND coalesce(node.summary, '') = ''
  AND NOT (node)-[:RELATES_TO]-() AND NOT (:Episodic)-[:MENTIONS]->(node)
DETACH DELETE node
"""

# The user's own statements, not what extraction made of them: an episode
# keeps «примерно в 2010 году» and the date it was said. The newest win the
# limit; the answer reads them in the order they were said.
PROFILE_EPISODES = """
MATCH (episode:Episodic {group_id: $namespace})
WHERE coalesce(episode.source_description, '') <> 'structured film export'
RETURN episode.uuid AS uuid, episode.name AS name, episode.content AS content,
       episode.valid_at AS valid_at
ORDER BY episode.valid_at DESC LIMIT $limit
"""

SET_EPISODE_EMBEDDING = (
    "MATCH (episode:Episodic {uuid: $uuid}) SET episode.content_embedding = $embedding"
)

RECENT_WATCHED_FILMS = (
    "MATCH (episode:Episodic {group_id: $namespace}) "
    "WHERE episode.source_description = 'structured film export' "
    "AND episode.watched_at IS NOT NULL "
    "RETURN episode.content AS content, episode.name AS name, "
    "episode.watched_at AS watched_at "
    "ORDER BY episode.watched_at DESC, episode.name ASC LIMIT $limit"
)

WATCHED_FILM_CONTENTS = (
    "MATCH (episode:Episodic {group_id: $namespace}) "
    "WHERE episode.source_description = 'structured film export' "
    "RETURN episode.content AS content"
)
