# Atomic Card Format

Atomic cards are JSON objects used by the retrieval system. Each card should represent one independent knowledge point and include evidence metadata.

## Common Fields

| Field | Description |
|---|---|
| `card_id` | Stable identifier, for example `fact_demo_001_001`. |
| `card_type` | `fact` or `case`. |
| `entity` | Canonical entity or concept name. |
| `title` | Short human-readable title. |
| `content` | Natural-language knowledge statement. |
| `structured_fields` | Type-specific structured information. |
| `retrieval_enhancement.keywords` | Optional keywords for retrieval. |
| `evidence.source_title` | Source title. |
| `evidence.page` | Source page. |
| `evidence.citation_text` | Short supporting excerpt. |

## Fact Example

```json
{
  "card_id": "fact_demo_001_001",
  "card_type": "fact",
  "entity": "Synthetic concept",
  "title": "Synthetic concept attribute",
  "content": "The synthetic concept has a demonstrative attribute.",
  "structured_fields": {
    "attribute_name": "attribute",
    "attribute_value": ["demonstrative"],
    "cold_heat": "neutral"
  },
  "retrieval_enhancement": {
    "keywords": ["synthetic concept", "attribute"]
  },
  "evidence": {
    "source_title": "Synthetic Source",
    "page": 1,
    "citation_text": "Synthetic supporting excerpt."
  }
}
```

## Case Example

```json
{
  "card_id": "case_demo_001_001",
  "card_type": "case",
  "entity": "Synthetic reasoning case",
  "title": "Synthetic condition-to-conclusion case",
  "content": "Given synthetic condition A and sign B, infer conclusion C.",
  "structured_fields": {
    "patient_conditions": {
      "symptoms": ["synthetic sign B"],
      "inducement": ["synthetic condition A"]
    },
    "reasoning_path": ["condition A", "sign B", "conclusion C"],
    "disease_nature": "neutral",
    "final_conclusion": "synthetic conclusion C"
  },
  "retrieval_enhancement": {
    "keywords": ["synthetic", "reasoning", "case"]
  },
  "evidence": {
    "source_title": "Synthetic Source",
    "page": 2,
    "citation_text": "Synthetic case excerpt."
  }
}
```
