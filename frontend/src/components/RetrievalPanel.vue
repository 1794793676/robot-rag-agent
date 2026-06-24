<script setup>
defineProps({
  toolCalls: { type: Array, default: () => [] },
  sources: { type: Array, default: () => [] },
})
</script>

<template>
  <section class="agent-card retrieval-panel">
    <div class="agent-card-head">
      <h2>工具与来源</h2>
    </div>
    <div class="tool-list">
      <article v-if="!toolCalls.length" class="tool-empty">暂无工具调用</article>
      <article v-for="item in toolCalls" :key="item.id" class="tool-item">
        <strong>{{ item.name }}</strong>
        <small>{{ item.status }}</small>
      </article>
    </div>
    <div v-if="sources.length" class="source-list compact">
      <article v-for="(source, index) in sources" :key="index" class="source">
        <header>
          <strong>{{ source.source || source.title }}</strong>
          <span>{{ source.score ? Number(source.score).toFixed(3) : source.source }}</span>
        </header>
        <p>{{ source.text || source.snippet }}</p>
      </article>
    </div>
  </section>
</template>

