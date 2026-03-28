# llm-tee

`llm-tee` is a T-shaped pipe for the Large Language Model era, inspired by the `tee` command.

- Like the traditional `tee` command, it allows you to capture data within a pipeline without affecting operations, enabling better observation of business data flow.
- It provides caching functionality, so you no longer consume expensive tokens for duplicate requests.
- It can replay modified client requests, allowing for faster debugging.
