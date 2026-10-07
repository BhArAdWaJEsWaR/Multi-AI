# Test Cases for Multi-Agent System

## 1. Model Persistence Tests

### Test 1.1: Model Reuse Within Same Domain
**Input:** 
```
You: write a hello world function in python
You: write a function to sort a list  
You: write a function to calculate fibonacci
```
**Expected:** 
- First query: `[coding/tier1 (qwen2.5-coder:1.5b) - LOADING]`
- Subsequent queries: `[coding/tier1 (qwen2.5-coder:1.5b) - REUSING LOADED]`
- Model should not be unloaded between similar coding queries

### Test 1.2: Topic Change and Model Unloading
**Input:**
```
You: write a python function to sort a list
You: what is 2 + 2
You: calculate 15 * 3
```
**Expected:**
- First query: `[coding/tier1 - LOADING]`
- Topic change message: `[model_registry] Topic changed from coding to math, unloading old domain models`
- Second query: `[math/tier1 - LOADING]` (new model loaded)
- Third query: `[math/tier1 - REUSING LOADED]` (math model reused)

### Test 1.3: Smart Tier Persistence
**Input:**
```
You: solve a simple math problem (easy)
You: solve a complex calculus problem (hard)
You: solve another simple math problem (easy)
```
**Expected:**
- If tier2/tier3 is already loaded for math, tier1 queries should reuse the higher-tier model
- Message: `[model_registry] Reusing loaded tier2 model for tier1 query`

## 2. Memory System Tests

### Test 2.1: Name Memory
**Input:**
```
You: what is my name
You: my name is bharadwaj
You: what is my name
```
**Expected:**
- First query: "I don't know your name"
- Second query: "Hello Bharadwaj! How can I assist you today?"
- Third query: "Your name is Bharadwaj" (memory system retrieves previous context)
- Memory context should show the Q&A pairs with the name

### Test 2.2: Preference Memory
**Input:**
```
You: i prefer recursion over iteration
You: write a function to sum array elements
```
**Expected:**
- First query: Should be classified as "general" and acknowledge preference
- Second query: Should provide a **recursive** solution, not iterative
- Memory context should contain the preference statement

### Test 2.3: Recency Weighting
**Input:**
```
You: what is my name (multiple times giving wrong answers)
You: my name is actually john
You: what is my name
```
**Expected:**
- System should prioritize the most recent "my name is john" over older incorrect responses
- Memory cleanup on startup should remove old "I don't know your name" entries

### Test 2.4: Multiple Information Types
**Input:**
```
You: my name is alice
You: i prefer dark mode
You: i work as a software engineer
You: what do you know about me
```
**Expected:**
- Should retrieve all three pieces of information
- Response should mention name, preference, and profession

## 3. Classification Tests

### Test 3.1: Preference Statement Detection
**Input:**
```
You: i prefer recursion over iteration
You: i like coffee
You: i want to learn python
```
**Expected:**
- All should be classified as "general"
- Should show: `[preference] Detected preference statement, classifying as general`
- Should NOT trigger coding/math responses

### Test 3.2: Coding vs General
**Input:**
```
You: write a function to sort a list (coding)
You: i prefer recursion (general preference)
You: what is recursion (general question)
```
**Expected:**
- First: `[coding]` with code solution
- Second: `[general]` acknowledging preference
- Third: `[general]` explaining recursion concept

### Test 3.3: Dataset-kNN Override Prevention
**Input:**
```
You: i prefer recursion methods (short preference)
```
**Expected:**
- Should NOT be overridden to "coding" by dataset-knn
- Should stay as "general" despite dataset containing recursion examples

## 4. Agent Behavior Tests

### Test 4.1: Validator Memory Awareness
**Input:**
```
You: my name is bob
You: what is my name
```
**Expected:**
- Tier1 agent: "Your name is Bob" (using memory)
- Validator should **NOT** flag this as LOW confidence
- Should understand that memory context was provided
- No tier escalation should occur

### Test 4.2: Coding Preference Following
**Input:**
```
You: i prefer recursion over iteration
You: Given an array of 1,000,000 integers, calculate the sum
```
**Expected:**
- Should provide recursive solution using helper function
- Should NOT use iterative for loop
- Solution should respect the preference from memory

### Test 4.3: General Agent Preference Acknowledgment
**Input:**
```
You: i prefer recursion over iteration
```
**Expected:**
- Response should acknowledge the preference
- Should confirm it will be remembered for future coding tasks
- Example: "Understood! I'll remember that you prefer recursion over iteration for coding tasks."

## 5. Edge Cases

### Test 5.1: Empty Memory
**Input:**
```
You: what is my name (fresh session, no prior context)
```
**Expected:**
- Should politely say it doesn't know the name
- Should not hallucinate a name
- Should ask user to provide their name

### Test 5.2: Conflicting Information
**Input:**
```
You: my name is alice
You: actually my name is bob
You: what is my name
```
**Expected:**
- Should use the most recent information (Bob)
- Recency weighting should prioritize the latest correction

### Test 5.3: Long Preference Statement
**Input:**
```
You: i would really prefer if you could use recursion instead of iteration when writing code for me because i find it more elegant
```
**Expected:**
- Should still be classified as general (preference)
- Should acknowledge the preference
- Should not trigger coding response

### Test 5.4: Mixed Domain Context
**Input:**
```
You: my name is charlie
You: write a sorting algorithm
You: what is my name
```
**Expected:**
- After switching to coding, should still remember name from general conversation
- Memory should persist across domain changes

## 6. Performance Tests

### Test 6.1: Memory Retrieval Speed
**Input:**
```
# Multiple quick queries testing memory speed
You: my name is test
You: what is my name
You: what is my name
You: what is my name
```
**Expected:**
- Memory retrieval should be fast (< 1 second)
- Subsequent "what is my name" queries should be quick due to model reuse

### Test 6.2: Model Loading Latency
**Input:**
```
You: write a hello world function (first coding query)
You: write another function (same domain)
```
**Expected:**
- First query: Slower (model loading)
- Second query: Faster (model reuse)
- Noticeable performance improvement

## 7. Integration Tests

### Test 7.1: Full Conversation Flow
**Input:**
```
You: hello
You: my name is david
You: i prefer recursion
You: write a fibonacci function
You: what is my name
You: write a sum function
```
**Expected:**
- Should remember name throughout
- Should use recursion for both coding tasks
- Should maintain context across domain switches
- Memory should work seamlessly with model persistence

### Test 7.2: Session Persistence
**Input:**
```
# First session
You: my name is emma
You: exit

# Second session (restart)
You: what is my name
```
**Expected:**
- Name should persist across sessions (ChromaDB is persistent)
- Should respond "Your name is Emma" on restart

## Manual Testing Commands

### Quick Test Sequence
```bash
python3 -m core.orchestrator
# Then run these commands in order:
1. what is my name
2. my name is [your name]
3. what is my name
4. i prefer recursion over iteration
5. write a function to sum array elements
6. what is my name
7. exit
```

### Model Persistence Test
```bash
python3 -m core.orchestrator
# Run these commands:
1. write a hello world function
2. write a sorting function
3. what is 2+2 (topic change)
4. write another function
# Check for "REUSING LOADED" vs "LOADING" messages
```

### Memory Cleanup Test
```bash
# Check startup output for:
[memory] Removed X entries containing 'I don't know your name'
# This confirms automatic memory cleanup is working
```

## Success Criteria

✅ **Model Persistence**: Models stay loaded and are reused appropriately
✅ **Memory System**: Names, preferences, and context are remembered correctly
✅ **Classification**: Preference statements are correctly identified as general
✅ **Agent Behavior**: Agents follow user preferences from memory
✅ **Validator Awareness**: Validators understand memory-provided context
✅ **Performance**: Noticeable speed improvement from model reuse
✅ **Cleanup**: Automatic removal of stale memory entries on startup