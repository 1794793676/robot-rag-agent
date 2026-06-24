<script setup>
defineProps({
  messages: { type: Array, default: () => [] },
})
</script>

<template>
  <section class="agent-card chat-panel">
    <div class="agent-card-head">
      <h2>对话</h2>
    </div>
    <div class="chat-list">
      <article v-if="!messages.length" class="chat-empty">等待语音或文字输入</article>
      <article v-for="message in messages" :key="message.id" class="chat-message" :class="message.role">
        <strong>{{ message.role === 'user' ? '你' : 'Agent' }}</strong>
        <p>{{ message.text }}</p>
      </article>
    </div>
  </section>
</template>

