// v2.0.72 (#115 #120) — NLU intent extractor tests.
//
// Cover real captures from probe runs against GLM-4.7 / GLM-5.1 / GPT-5.5 /
// Kimi-K2 in cascade backend, plus synthetic patterns we expect future
// models to use. Layer 1 (explicit syntax) → Layer 3 (narrative) ordered
// by confidence.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { extractIntentFromNarrative } from '../src/handlers/intent-extractor.js';

const fnTool = (name, props = { command: 'string' }, required = ['command']) => ({
  type: 'function',
  function: {
    name,
    description: `${name} description`,
    parameters: {
      type: 'object',
      properties: Object.fromEntries(Object.entries(props).map(([k, t]) => [k, { type: t }])),
      required,
    },
  },
});

const SHELL_TOOL = fnTool('shell_exec');
const BASH_TOOL = fnTool('Bash');
const READ_TOOL = fnTool('Read', { file_path: 'string' }, ['file_path']);
const WRITE_TOOL = fnTool('Write', { file_path: 'string', content: 'string' }, ['file_path', 'content']);
const ACTIONABLE = { lastUserText: 'run shell_exec to echo something' };

describe('Layer 1 — explicit invocation syntax', () => {
  it('extracts shell_exec(command="echo HI")', () => {
    const r = extractIntentFromNarrative(
      'shell_exec(command="echo HI")',
      [SHELL_TOOL], ACTIONABLE,
    );
    assert.equal(r.length, 1);
    assert.equal(r[0].name, 'shell_exec');
    assert.deepEqual(JSON.parse(r[0].argumentsJson), { command: 'echo HI' });
    assert.equal(r[0].layer, 'explicit-syntax');
    assert.ok(r[0].confidence >= 0.9);
  });

  it('extracts function_call: name=shell_exec args={"command":"X"}', () => {
    const r = extractIntentFromNarrative(
      'function_call: name=shell_exec args={"command":"echo X"}',
      [SHELL_TOOL], ACTIONABLE,
    );
    assert.equal(r.length, 1);
    assert.equal(r[0].name, 'shell_exec');
    assert.equal(JSON.parse(r[0].argumentsJson).command, 'echo X');
  });

  it('rejects fn name not in tools[]', () => {
    const r = extractIntentFromNarrative(
      'os_command(cmd="echo X")', [SHELL_TOOL], ACTIONABLE,
    );
    assert.equal(r.length, 0);
  });
});

describe('Layer 2 — backtick-quoted name + value', () => {
  it("extracts I'll call `shell_exec` with command `echo HI`", () => {
    const r = extractIntentFromNarrative(
      "I'll call `shell_exec` with command `echo HI`",
      [SHELL_TOOL], ACTIONABLE,
    );
    assert.equal(r.length, 1);
    assert.equal(r[0].layer, 'backtick-quoted');
    assert.deepEqual(JSON.parse(r[0].argumentsJson), { command: 'echo HI' });
  });

  it('extracts use the `Read` function with file_path `/etc/hosts`', () => {
    const r = extractIntentFromNarrative(
      'use the `Read` function with file_path `/etc/hosts`',
      [READ_TOOL], { lastUserText: 'read the file at /etc/hosts' },
    );
    assert.equal(r.length, 1);
    assert.equal(r[0].name, 'Read');
    assert.deepEqual(JSON.parse(r[0].argumentsJson), { file_path: '/etc/hosts' });
  });
});

describe('Layer 3 — natural narrative (live GLM-4.7 reproducer)', () => {
  it("LIVE: 'I should call the shell_exec function with the command \"echo HELLO_FROM_PROBE\"'", () => {
    // This is the actual emit captured from glm-4.7 probe before v2.0.72.
    const r = extractIntentFromNarrative(
      'I should call the shell_exec function with the command "echo HELLO_FROM_PROBE".',
      [SHELL_TOOL], ACTIONABLE,
    );
    assert.equal(r.length, 1);
    assert.equal(r[0].name, 'shell_exec');
    assert.deepEqual(JSON.parse(r[0].argumentsJson), { command: 'echo HELLO_FROM_PROBE' });
    assert.equal(r[0].layer, 'narrative');
  });

  it("'Let me run shell_exec with command echo HI'", () => {
    const r = extractIntentFromNarrative(
      "Let me run shell_exec with command 'echo HI'",
      [SHELL_TOOL], ACTIONABLE,
    );
    assert.equal(r.length, 1);
    assert.equal(r[0].name, 'shell_exec');
  });

  it('recovers Claude Code thinking narration with a quoted Bash command', () => {
    const r = extractIntentFromNarrative(
      '我只需要：\n1. 调用 Bash 工具执行 `printf "TOOLCALL_OK\\n"`\n2. 等待工具执行完成',
      [BASH_TOOL],
      { lastUserText: '你必须调用 Bash 工具执行这个精确命令：printf "TOOLCALL_OK\\n"' },
    );
    assert.equal(r.length, 1);
    assert.equal(r[0].name, 'Bash');
    assert.deepEqual(JSON.parse(r[0].argumentsJson), { command: 'printf "TOOLCALL_OK\\n"' });
  });

  it('recovers Claude Code numbered Chinese plan with an unquoted Bash command', () => {
    const r = extractIntentFromNarrative(
      '让我按照步骤执行：\n1. 调用 Bash 工具执行：printf "tool-ok\\n"\n2. 新建 generated.txt',
      [BASH_TOOL],
      { lastUserText: '调用 Bash 工具执行：printf "tool-ok\\n"' },
    );
    assert.equal(r.length, 1);
    assert.equal(r[0].name, 'Bash');
    assert.deepEqual(JSON.parse(r[0].argumentsJson), { command: 'printf "tool-ok\\n"' });
  });

  it('recovers Claude Code numbered plan lines like Bash: command', () => {
    const r = extractIntentFromNarrative(
      '步骤：\n1. Bash: printf "tool-ok\\n"\n2. Bash: echo "created-by-claude-code" > generated.txt',
      [BASH_TOOL],
      { lastUserText: '调用 Bash 工具创建文件并输出 tool-ok' },
    );
    assert.deepEqual(
      r.map(x => JSON.parse(x.argumentsJson).command),
      ['printf "tool-ok\\n"', 'echo "created-by-claude-code" > generated.txt'],
    );
  });

  it('recovers concrete shell commands from Claude Code step labels', () => {
    const r = extractIntentFromNarrative(
      '步骤1: printf "tool-ok\\n"\n步骤2: printf "created-by-claude-code\\n" > generated.txt\n步骤3: printf "before-edit\\n" > edit-target.txt',
      [BASH_TOOL],
      { lastUserText: '用 Bash 创建 generated.txt 和 edit-target.txt' },
    );
    assert.deepEqual(
      r.map(x => JSON.parse(x.argumentsJson).command),
      [
        'printf "tool-ok\\n"',
        'printf "created-by-claude-code\\n" > generated.txt',
        'printf "before-edit\\n" > edit-target.txt',
      ],
    );
  });

  it('strips Claude Code Bash command labels before executing', () => {
    const r = extractIntentFromNarrative(
      '1. Bash command: printf "created-by-claude-code\\n" > generated.txt',
      [BASH_TOOL],
      { lastUserText: 'Bash command: printf "created-by-claude-code\\n" > generated.txt' },
    );
    assert.equal(r.length, 1);
    assert.deepEqual(JSON.parse(r[0].argumentsJson), {
      command: 'printf "created-by-claude-code\\n" > generated.txt',
    });
  });

  it('does not recover already-completed Bash steps', () => {
    const r = extractIntentFromNarrative(
      '1. 调用 Bash 工具执行：printf "tool-ok\\n" - 已经完成',
      [BASH_TOOL],
      { lastUserText: '调用 Bash 工具执行：printf "tool-ok\\n"' },
    );
    assert.equal(r.length, 0);
  });

  it('does not recover Bash history lines marked simply as 完成', () => {
    const r = extractIntentFromNarrative(
      '1. Bash command: printf "tool-ok\\n" - 完成\n现在执行步骤4：Read file_path: edit-target.txt',
      [BASH_TOOL, READ_TOOL],
      { lastUserText: '分轮执行 Bash 和 Read' },
    );
    assert.deepEqual(r.map(x => [x.name, JSON.parse(x.argumentsJson)]), [
      ['Read', { file_path: 'edit-target.txt' }],
    ]);
  });

  it('does not treat completed step ranges as the active range', () => {
    const r = extractIntentFromNarrative(
      '我已经完成了步骤1-3：\n'
        + '1. Bash command: printf "tool-ok\\n" - 完成\n'
        + '2. Bash command: printf "created-by-claude-code\\n" > generated.txt - 完成\n'
        + '3. Bash command: printf "before-edit\\n" > edit-target.txt - 完成\n'
        + '现在收到了步骤1-3的结果，根据指示，接下来应该执行：\n'
        + '4. Read file_path: edit-target.txt',
      [BASH_TOOL, READ_TOOL],
      { lastUserText: '分轮执行 Bash 和 Read' },
    );
    assert.deepEqual(r.map(x => [x.name, JSON.parse(x.argumentsJson)]), [
      ['Read', { file_path: 'edit-target.txt' }],
    ]);
  });

  it('recovers Edit old/new arguments from Claude Code Chinese plan text', () => {
    const editTool = { type: 'function', function: { name: 'Edit', parameters: { type: 'object', properties: { file_path: { type: 'string' }, old_string: { type: 'string' }, new_string: { type: 'string' } } } } };
    const r = extractIntentFromNarrative(
      '使用 Edit 工具把 edit-target.txt 中的 before-edit 修改为 after-edit',
      [editTool],
      { lastUserText: '使用 Edit 工具把 edit-target.txt 中的 before-edit 修改为 after-edit' },
    );
    assert.equal(r.length, 1);
    assert.deepEqual(JSON.parse(r[0].argumentsJson), {
      file_path: 'edit-target.txt',
      old_string: 'before-edit',
      new_string: 'after-edit',
    });
  });

  it('recovers Edit and Read field-label plan lines', () => {
    const editTool = { type: 'function', function: { name: 'Edit', parameters: { type: 'object', properties: { file_path: { type: 'string' }, old_string: { type: 'string' }, new_string: { type: 'string' } } } } };
    const readTool = { type: 'function', function: { name: 'Read', parameters: { type: 'object', properties: { file_path: { type: 'string' } } } } };
    const r = extractIntentFromNarrative(
      '4. Edit file_path: edit-target.txt, old_string: before-edit, new_string: after-edit\n5. Read file_path: generated.txt\n6. Read file_path: edit-target.txt',
      [editTool, readTool],
      { lastUserText: 'Edit file_path: edit-target.txt old_string: before-edit new_string: after-edit Read file_path: generated.txt' },
    );
    assert.deepEqual(r.map(x => [x.name, JSON.parse(x.argumentsJson)]), [
      ['Edit', { file_path: 'edit-target.txt', old_string: 'before-edit', new_string: 'after-edit' }],
      ['Read', { file_path: 'generated.txt' }],
      ['Read', { file_path: 'edit-target.txt' }],
    ]);
  });

  it('recovers Claude Code core-tool English plan in source order', () => {
    const editTool = fnTool(
      'Edit',
      { file_path: 'string', old_string: 'string', new_string: 'string' },
      ['file_path', 'old_string', 'new_string'],
    );
    const text = `The user wants me to perform a sequence of tool operations in order:

1. Use Read tool to read read-target.txt
2. Use Edit tool to change the unique "before-edit" in edit-target.txt to "after-edit"
3. Use Write tool to create generated.txt with content "created-by-write-tool"
4. Use Bash tool to execute: printf 'bash-ok'
5. Use Read or Bash to confirm generated.txt and edit-target.txt content`;
    const r = extractIntentFromNarrative(
      text,
      [READ_TOOL, editTool, WRITE_TOOL, BASH_TOOL],
      { lastUserText: 'Use Read, Edit, Write, Bash tools to create and edit files' },
    );
    assert.deepEqual(r.map(x => [x.name, JSON.parse(x.argumentsJson)]), [
      ['Read', { file_path: 'read-target.txt' }],
      ['Edit', { file_path: 'edit-target.txt', old_string: 'before-edit', new_string: 'after-edit' }],
      ['Write', { file_path: 'generated.txt', content: 'created-by-write-tool' }],
      ['Bash', { command: "printf 'bash-ok'" }],
    ]);
  });

  it('recovers Write from Claude Code thinking narration', () => {
    const r = extractIntentFromNarrative(
      '用户要求我使用 Write 工具创建一个名为 generated.txt 的文件，内容为 "write-tool-ok"。让我执行这个任务。',
      [WRITE_TOOL],
      { lastUserText: 'Use the Write tool to create generated.txt with exactly: write-tool-ok' },
    );
    assert.deepEqual(r.map(x => [x.name, JSON.parse(x.argumentsJson)]), [
      ['Write', { file_path: 'generated.txt', content: 'write-tool-ok' }],
    ]);
  });

  it('recovers Chinese Claude Code core-tool plan and skips completed steps', () => {
    const editTool = fnTool(
      'Edit',
      { file_path: 'string', old_string: 'string', new_string: 'string' },
      ['file_path', 'old_string', 'new_string'],
    );
    const text = `用户要求我按顺序完成以下任务：
1. 使用 Read 工具读取 read-target.txt - 已完成，内容是 "read-ok"
2. 使用 Edit 工具把 edit-target.txt 中唯一的 before-edit 改成 after-edit - 需要执行
3. 使用 Write 工具新建 generated.txt，内容只写一行：created-by-write-tool - 需要执行
4. 使用 Bash 工具执行：printf 'bash-ok' - 已完成，输出是 bash-ok`;
    const r = extractIntentFromNarrative(
      text,
      [READ_TOOL, editTool, WRITE_TOOL, BASH_TOOL],
      { lastUserText: '使用 Read、Edit、Write、Bash 工具创建和编辑文件' },
    );
    assert.deepEqual(r.map(x => [x.name, JSON.parse(x.argumentsJson)]), [
      ['Edit', { file_path: 'edit-target.txt', old_string: 'before-edit', new_string: 'after-edit' }],
      ['Write', { file_path: 'generated.txt', content: 'created-by-write-tool' }],
    ]);
  });

  it('does not fabricate partial Edit calls from generic retry prose', () => {
    const editTool = fnTool(
      'Edit',
      { file_path: 'string', old_string: 'string', new_string: 'string' },
      ['file_path', 'old_string', 'new_string'],
    );
    const r = extractIntentFromNarrative(
      '所以问题是 Edit 工具要求先读取文件才能编辑。我需要先读取 edit-target.txt，然后再执行 Edit 操作。',
      [editTool, READ_TOOL],
      { lastUserText: '使用 Edit 工具把 edit-target.txt 中的 before-edit 修改为 after-edit' },
    );
    assert.deepEqual(r.map(x => [x.name, JSON.parse(x.argumentsJson)]), [
      ['Read', { file_path: 'edit-target.txt' }],
    ]);
  });

  it('honours Claude Code current-step range before promoting plan lines', () => {
    const editTool = fnTool(
      'Edit',
      { file_path: 'string', old_string: 'string', new_string: 'string' },
      ['file_path', 'old_string', 'new_string'],
    );
    const text = `我需要按照这些步骤执行。
1. Bash command: printf "tool-ok\\n"
2. Bash command: printf "created-by-claude-code\\n" > generated.txt
3. Bash command: printf "before-edit\\n" > edit-target.txt
4. Edit file_path: edit-target.txt, old_string: before-edit, new_string: after-edit
5. Read file_path: generated.txt
6. Read file_path: edit-target.txt

让我先并行执行步骤1-3。`;
    const r = extractIntentFromNarrative(
      text,
      [BASH_TOOL, editTool, READ_TOOL],
      { lastUserText: '真实调用工具创建并编辑文件' },
    );
    assert.deepEqual(r.map(x => [x.name, JSON.parse(x.argumentsJson)]), [
      ['Bash', { command: 'printf "tool-ok\\n"' }],
      ['Bash', { command: 'printf "created-by-claude-code\\n" > generated.txt' }],
      ['Bash', { command: 'printf "before-edit\\n" > edit-target.txt' }],
    ]);
  });

  it('honours Claude Code current single-step intent before promoting plan lines', () => {
    const editTool = fnTool(
      'Edit',
      { file_path: 'string', old_string: 'string', new_string: 'string' },
      ['file_path', 'old_string', 'new_string'],
    );
    const text = `1. Read file_path: edit-target.txt
2. Edit file_path: edit-target.txt, old_string: before-edit, new_string: after-edit
3. Read file_path: edit-target.txt

接下来执行步骤2。`;
    const r = extractIntentFromNarrative(
      text,
      [editTool, READ_TOOL],
      { lastUserText: '使用 Edit 工具把 edit-target.txt 中的 before-edit 修改为 after-edit' },
    );
    assert.deepEqual(r.map(x => [x.name, JSON.parse(x.argumentsJson)]), [
      ['Edit', { file_path: 'edit-target.txt', old_string: 'before-edit', new_string: 'after-edit' }],
    ]);
  });

  it('skips completed Read history before recovering the current Edit step', () => {
    const editTool = fnTool(
      'Edit',
      { file_path: 'string', old_string: 'string', new_string: 'string' },
      ['file_path', 'old_string', 'new_string'],
    );
    const text = `用户要求我必须分轮执行，不能提前调用未来步骤。现在我已经完成了步骤1-3：
1. Bash command: printf "tool-ok\\n" - 完成，输出 "tool-ok"
2. Bash command: printf "created-by-claude-code\\n" > generated.txt - 完成
3. Bash command: printf "before-edit\\n" > edit-target.txt - 完成

然后我执行了步骤4：
4. Read file_path: edit-target.txt - 完成，内容是 "before-edit"

现在根据指示，收到步骤4结果后，再执行步骤5：
5. Edit file_path: edit-target.txt, old_string: before-edit, new_string: after-edit

我需要执行这个Edit操作。`;
    const r = extractIntentFromNarrative(
      text,
      [BASH_TOOL, editTool, READ_TOOL],
      { lastUserText: '真实调用工具创建、读取并编辑文件' },
    );
    assert.deepEqual(r.map(x => [x.name, JSON.parse(x.argumentsJson)]), [
      ['Edit', { file_path: 'edit-target.txt', old_string: 'before-edit', new_string: 'after-edit' }],
    ]);
  });

  it('keeps inline active-step lines before recovering Edit', () => {
    const editTool = fnTool(
      'Edit',
      { file_path: 'string', old_string: 'string', new_string: 'string' },
      ['file_path', 'old_string', 'new_string'],
    );
    const text = `现在已经完成了步骤1-3，收到了步骤4的结果（读取edit-target.txt文件，内容是"before-edit"）。

根据指令，接下来应该执行步骤5：Edit file_path: edit-target.txt, old_string: before-edit, new_string: after-edit

然后收到步骤5结果后，再执行步骤6和7：Read generated.txt 和 Read edit-target.txt

现在执行步骤5。`;
    const r = extractIntentFromNarrative(
      text,
      [BASH_TOOL, editTool, READ_TOOL],
      { lastUserText: '真实调用工具创建、读取并编辑文件' },
    );
    assert.deepEqual(r.map(x => [x.name, JSON.parse(x.argumentsJson)]), [
      ['Edit', { file_path: 'edit-target.txt', old_string: 'before-edit', new_string: 'after-edit' }],
    ]);
  });

  it("'I'll invoke the Read tool to read /etc/hosts'", () => {
    const r = extractIntentFromNarrative(
      "I'll invoke the Read tool to read /etc/hosts",
      [READ_TOOL], { lastUserText: 'read /etc/hosts' },
    );
    assert.equal(r.length, 1);
    assert.equal(r[0].name, 'Read');
    assert.deepEqual(JSON.parse(r[0].argumentsJson), { file_path: '/etc/hosts' });
  });

  it('Layer 3 only fires when user prompt is actionable', () => {
    const r = extractIntentFromNarrative(
      'I should call the shell_exec function with the command "echo HI".',
      [SHELL_TOOL],
      { lastUserText: 'tell me about your day' }, // NOT actionable
    );
    assert.equal(r.length, 0);
  });

  // v2.0.76 follow-up — caught in v2.0.75 e2e probe against glm-4.7.
  // GLM emitted "...with command 'command'" (the literal word) which
  // made the regex bind value="command". Filter placeholder values.
  it("rejects placeholder values ('command' / 'argument' / 'input' / etc.)", () => {
    const r = extractIntentFromNarrative(
      "I'll call shell_exec with command 'command'.",
      [SHELL_TOOL], ACTIONABLE,
    );
    assert.equal(r.length, 0);
  });

  it("dedupes when narrative says the real command then echoes 'with command command'", () => {
    // Real GLM-4.7 v2.0.75 probe pattern that produced 2 tool_calls,
    // one valid and one bogus. Now should produce just 1.
    const r = extractIntentFromNarrative(
      `I'll call shell_exec with command 'echo HELLO'. The user wants me to use the shell_exec function with command 'command' as the parameter name.`,
      [SHELL_TOOL], ACTIONABLE,
    );
    assert.equal(r.length, 1);
    assert.deepEqual(JSON.parse(r[0].argumentsJson), { command: 'echo HELLO' });
  });
});

describe('robustness', () => {
  it('returns [] for hopeless fabricated output (just a number)', () => {
    const r = extractIntentFromNarrative('1777751588', [SHELL_TOOL], ACTIONABLE);
    assert.equal(r.length, 0);
  });

  it('returns [] for empty / null input', () => {
    assert.deepEqual(extractIntentFromNarrative('', [SHELL_TOOL], ACTIONABLE), []);
    assert.deepEqual(extractIntentFromNarrative(null, [SHELL_TOOL], ACTIONABLE), []);
    assert.deepEqual(extractIntentFromNarrative('text', [], ACTIONABLE), []);
  });

  it('dedupes identical extractions', () => {
    const text = 'I should call the shell_exec function with the command "echo X". '
      + 'shell_exec(command="echo X")';
    const r = extractIntentFromNarrative(text, [SHELL_TOOL], ACTIONABLE);
    // Same (name, args) → 1 entry. Layer 1 wins on confidence.
    assert.equal(r.length, 1);
    assert.equal(r[0].layer, 'explicit-syntax');
  });

  it('keeps multiple distinct tool_calls', () => {
    const text = 'shell_exec(command="ls")\nshell_exec(command="pwd")';
    const r = extractIntentFromNarrative(text, [SHELL_TOOL], ACTIONABLE);
    assert.equal(r.length, 2);
    const cmds = r.map(x => JSON.parse(x.argumentsJson).command).sort();
    assert.deepEqual(cmds, ['ls', 'pwd']);
  });

  it('env WINDSURFAPI_NLU_RECOVERY=0 disables extractor entirely', () => {
    const orig = process.env.WINDSURFAPI_NLU_RECOVERY;
    process.env.WINDSURFAPI_NLU_RECOVERY = '0';
    try {
      const r = extractIntentFromNarrative(
        'shell_exec(command="echo HI")', [SHELL_TOOL], ACTIONABLE,
      );
      assert.equal(r.length, 0);
    } finally {
      if (orig !== undefined) process.env.WINDSURFAPI_NLU_RECOVERY = orig;
      else delete process.env.WINDSURFAPI_NLU_RECOVERY;
    }
  });
});

describe('confidence threshold opt', () => {
  it('opt.minConfidence filters layer 3 narrative-only extractions', () => {
    // Default threshold lets narrative through (0.65). Bump to 0.8
    // and only Layer 1+2 survive.
    const text = 'I should call the shell_exec function with the command "echo HI".';
    const high = extractIntentFromNarrative(text, [SHELL_TOOL], { ...ACTIONABLE, minConfidence: 0.8 });
    assert.equal(high.length, 0);
    const low = extractIntentFromNarrative(text, [SHELL_TOOL], { ...ACTIONABLE, minConfidence: 0.5 });
    assert.equal(low.length, 1);
  });
});
