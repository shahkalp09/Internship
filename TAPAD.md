**TAPAS (TAble PArSing):**

**Introduction:**

TAPAS extends the BERT architecture to encode the question jointly along with tabular data structure, resulting in a model that can then point directly to the answer. Instead of creating a model that works only for a single style of table, this approach results in a model that can be applied to tables from a wide range of domains.

**TAPAS Model:**

Our model's architecture is based on BERT's encoder with additional positional embeddings used to encode tabular structure (visualized in Figure 2). We flatten the table into a sequence of words, split words into word pieces (tokens) and concatenate the question tokens before the table tokens.

We use different kinds of positional embeddings:

- Position ID is the index of the token in the flattened sequence (same as in BERT).
- Segment ID takes two possible values: 0 for the question, and 1 for the table header and cells.
- Column / Row ID is the index of the column/row that this token appears in, or 0 if the token is a part of the question.
- Rank ID if column values can be parsed as floats or dates, we sort them accordingly and assign an embedding based on their numeric rank (0 for not comparable, 1 for the smallest item, i + 1 for an item with rank i). This can assist the model when processing questions that involve superlatives, as word pieces may not represent numbers informatively
- Previous Answer given a conversational setup where the current question might refer to the previous question or its answers, we add a special embedding that marks whether a cell token was the answer to the previous question (1 if the token's cell was an answer or 0 otherwise).

![](RackMultipart20230609-1-hnxx25_html_7505dd8641d0dd6c.png)

**Cell Selection:**

The classification layer selects the subset of the cell from the table according to requirements. Depending on the selected aggregation operator, these cells can be the final answer or the input used to compute the final answer. Cells are modeled as independent Bernoulli variables. First, we compute the logit for a token using a linear layer on top of its last hidden vector. Cell logits are then computed as the average over logits of tokens in that cell. The output of the layer is the probability p(c)s to select cell c.

![](RackMultipart20230609-1-hnxx25_html_942af99d4b2b4bef.png)

**Aggregation operator prediction:**

Semantic parsing tasks require discrete reasoning over the table, such as summing numbers or counting cells. To handle these cases without producing logical forms, TAPAS outputs a subset of the table cells together with an optional aggregation operator. The aggregation operator describes an operation to be applied to the selected cells, such as SUM, COUNT, AVERAGE or NONE. The operator is selected by a linear layer followed by a softmax on top of the final hidden vector of the first token (the special [CLS] token). We denote this layer as pa(op), where op is some aggregation operator.

**Inference:**

We predict the most likely aggregation operator together with a subset of the cells (using the cell selection layer). We select all table cells for which their probability is larger than 0.5 to predict a discrete cell selection. These predictions are then executed against the table to retrieve the answer, by applying the predicted aggregation over the selected cells.

**Limitations:**

1. TAPAS handles single tables as context, which are able to fit in memory. Thus, TAPAS model would fail to capture very large tables, or databases that contain multiple tables.
2. Although TAPAS can parse compositional structures, its expressivity is limited to a form of aggregation over a subset of table cells. Thus, structures with multiple aggregations such as "number of actors with an average rating higher than 4" could not be handled correctly.

**References:**

Understanding tables with intermediate pre-training

(Julian Eisenschlos, Syrine Krichene, Thomas Müller)
