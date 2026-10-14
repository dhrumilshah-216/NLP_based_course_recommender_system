# My College: Synthetic NLP Course Recommendation Dataset

## Purpose
A prototype dataset for an NLP mini project combining semantic course search, intent extraction, aspect-based sentiment analysis, review summarization, and personalized ranking.

## Important data provenance notice
- Institution: fictional "My College".
- All course records and course-level attributes in `courses.csv` are synthetic and reference-inspired.
- All reviews, ratings, student profiles, review-aspect labels, and expected query relevance labels are synthetic.
- No review represents a real student, alumnus, or faculty member.
- Do not claim the synthetic course scores reflect actual difficulty, grades, workload, or teaching quality.
- Assessment splits and credit formats are illustrative and are not official policy.

## Files
- `courses.csv`: 29 fictional course records.
- `reviews.csv`: 870 synthetic reviews (30 per course).
- `review_aspects.csv`: one row per review/aspect label, suitable for ABSA experiments.
- `student_profiles.csv`: 10 synthetic student preference profiles.
- `queries_and_relevance.csv`: 20 synthetic natural-language queries and expected relevant course IDs.
- `course_synthetic_aggregates.csv`: aggregates computed from synthetic review labels only.
- `data_dictionary.csv`: key fields and meanings.
- `dataset_summary.json`: dataset counts and limitations.

## Recommended first experiments
1. Use `courses.csv` to build a semantic index over `course_title + syllabus_topics + tools_technologies + topic_cluster`.
2. Use `queries_and_relevance.csv` to smoke-test query handling and compare semantic-only versus hybrid ranking. Treat relevance labels as illustrative and manually review them.
3. Use `review_aspects.csv` as controlled labels for testing an aspect/sentiment extraction pipeline. Since labels are synthetic, this tests consistency on synthetic examples, not real-world generalization.
4. Use `reviews.csv` for display and summarization demos, with a visible synthetic-data notice.
5. Keep any held-out split grouped by review or course as appropriate. Avoid having near-identical templates from the same course in both train and test.
6. Never use the synthetic aggregates as evidence of actual student experiences.

## Known limitations
- Review language is generated from controlled templates, so it may be less diverse than genuine student writing.
- Synthetic labels are not independent human annotations.
- Course attributes are illustrative, not institutionally verified.
- Expected query-course relevance labels are hypotheses for testing and should be manually reviewed.
