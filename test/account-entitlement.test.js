import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { isModelAllowedForAccount, getAvailableModelsForAccount } from '../src/auth.js';
import { MODELS } from '../src/models.js';

// Free Windsurf accounts entitled by `cascade_allowed_models_config` to
// GLM/SWE/Kimi were getting routed away from the proxy because
// `MODEL_TIER_ACCESS.free` is a static `['gemini-2.5-flash', ...]` list
// that ignored per-account capabilities populated authoritatively by
// GetUserStatus. The fix is to trust `capabilities[key].reason ===
// 'user_status'` over the tier list once GetUserStatus has run.

describe('isModelAllowedForAccount — capabilities-first routing', () => {
  it('honours user_status capability for free accounts entitled to GLM', () => {
    const account = {
      tier: 'free',
      userStatusLastFetched: Date.now(),
      capabilities: {
        'glm-4.7': { ok: true, reason: 'user_status', lastCheck: 1 },
        'gemini-2.5-flash': { ok: true, reason: 'user_status', lastCheck: 1 },
        'claude-opus-4.6': { ok: false, reason: 'not_entitled', lastCheck: 1 },
      },
    };
    assert.equal(isModelAllowedForAccount(account, 'glm-4.7'), true);
    assert.equal(isModelAllowedForAccount(account, 'gemini-2.5-flash'), true);
    assert.equal(isModelAllowedForAccount(account, 'claude-opus-4.6'), false);
  });

  it('blocks pro-only models even on a Pro account when not_entitled', () => {
    const account = {
      tier: 'pro',
      userStatusLastFetched: Date.now(),
      capabilities: {
        'claude-opus-4.6': { ok: true, reason: 'user_status', lastCheck: 1 },
        'glm-4.7': { ok: false, reason: 'not_entitled', lastCheck: 1 },
      },
    };
    assert.equal(isModelAllowedForAccount(account, 'glm-4.7'), false);
  });

  it('respects blocklist regardless of upstream entitlement', () => {
    const account = {
      tier: 'free',
      blockedModels: ['glm-4.7'],
      userStatusLastFetched: Date.now(),
      capabilities: {
        'glm-4.7': { ok: true, reason: 'user_status', lastCheck: 1 },
      },
    };
    assert.equal(isModelAllowedForAccount(account, 'glm-4.7'), false);
  });

  it('falls back to tier list when GetUserStatus has not run', () => {
    const account = { tier: 'free', capabilities: {} };
    assert.equal(isModelAllowedForAccount(account, 'gemini-2.5-flash'), true);
    assert.equal(isModelAllowedForAccount(account, 'glm-4.7'), false);
  });

  it('falls back to tier list for capabilities filled by canary success', () => {
    const account = {
      tier: 'free',
      capabilities: {
        'gemini-2.5-flash': { ok: true, reason: 'success', lastCheck: 1 },
      },
    };
    assert.equal(isModelAllowedForAccount(account, 'gemini-2.5-flash'), true);
  });

  it('honours persisted success for dynamic UID-only models after restart', () => {
    const original = MODELS['swe-1-6-slow'];
    MODELS['swe-1-6-slow'] = {
      name: 'swe-1-6-slow',
      provider: 'windsurf',
      enumValue: 0,
      modelUid: 'swe-1-6-slow',
      credit: 1,
    };
    try {
      const account = {
        tier: 'free',
        userStatusLastFetched: Date.now(),
        capabilities: {
          'swe-1-6-slow': { ok: true, reason: 'success', lastCheck: 1 },
        },
      };
      assert.equal(isModelAllowedForAccount(account, 'swe-1-6-slow'), true);
    } finally {
      if (original) MODELS['swe-1-6-slow'] = original;
      else delete MODELS['swe-1-6-slow'];
    }
  });

  it('manual tier=pro override unlocks all models even with not_entitled caps', () => {
    // Operator escape hatch: probe misclassified a Pro trial as free,
    // GetUserStatus then wrote not_entitled into every premium model's
    // capability slot. Operator manually sets tier=pro; that should
    // restore Pro entitlement until GetUserStatus reruns and corrects
    // capabilities itself.
    const account = {
      tier: 'pro',
      tierManual: true,
      userStatusLastFetched: Date.now(),
      capabilities: {
        'claude-opus-4.6': { ok: false, reason: 'not_entitled', lastCheck: 1 },
        'glm-4.7': { ok: false, reason: 'not_entitled', lastCheck: 1 },
      },
    };
    assert.equal(isModelAllowedForAccount(account, 'claude-opus-4.6'), true);
    assert.equal(isModelAllowedForAccount(account, 'glm-4.7'), true);
  });
});

describe('getAvailableModelsForAccount — uses authoritative allowlist post-status', () => {
  it('returns only user_status-allowed enum-keyed models after GetUserStatus', () => {
    const account = {
      tier: 'free',
      userStatusLastFetched: Date.now(),
      capabilities: {
        'gemini-2.5-flash': { ok: true, reason: 'user_status', lastCheck: 1 },
        'glm-4.7': { ok: true, reason: 'user_status', lastCheck: 1 },
        'kimi-k2': { ok: true, reason: 'user_status', lastCheck: 1 },
        'swe-1.5': { ok: true, reason: 'user_status', lastCheck: 1 },
        'claude-opus-4.6': { ok: false, reason: 'not_entitled', lastCheck: 1 },
        'gpt-4.1-mini': { ok: false, reason: 'not_entitled', lastCheck: 1 },
      },
    };
    const got = getAvailableModelsForAccount(account);
    assert.ok(got.includes('gemini-2.5-flash'));
    assert.ok(got.includes('glm-4.7'));
    assert.ok(got.includes('kimi-k2'));
    assert.ok(got.includes('swe-1.5'));
    assert.ok(!got.includes('claude-opus-4.6'));
    assert.ok(!got.includes('gpt-4.1-mini'));
  });

  it('returns persisted successful dynamic models and hides deprecated entries', () => {
    const original = MODELS['swe-1-6-slow'];
    MODELS['swe-1-6-slow'] = {
      name: 'swe-1-6-slow',
      provider: 'windsurf',
      enumValue: 0,
      modelUid: 'swe-1-6-slow',
      credit: 1,
    };
    try {
      const account = {
        tier: 'free',
        userStatusLastFetched: Date.now(),
        capabilities: {
          'swe-1-6-slow': { ok: true, reason: 'success', lastCheck: 1 },
          'gpt-4o-mini': { ok: true, reason: 'user_status', lastCheck: 1 },
          'gemini-2.5-flash': { ok: true, reason: 'user_status', lastCheck: 1 },
        },
      };
      const got = getAvailableModelsForAccount(account);
      assert.ok(got.includes('swe-1-6-slow'));
      assert.ok(got.includes('gemini-2.5-flash'));
      assert.ok(!got.includes('gpt-4o-mini'));
    } finally {
      if (original) MODELS['swe-1-6-slow'] = original;
      else delete MODELS['swe-1-6-slow'];
    }
  });

  it('falls back to tier list before GetUserStatus runs', () => {
    const account = { tier: 'free' };
    const got = getAvailableModelsForAccount(account);
    assert.ok(got.includes('gemini-2.5-flash'));
    assert.ok(!got.includes('glm-4.7'));
  });
});
