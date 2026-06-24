<script setup>
defineProps({
  active: Boolean,
  disabled: Boolean,
})
defineEmits(['click'])
</script>

<template>
  <button class="voice-button" :class="{ active }" :disabled="disabled" @click="$emit('click')">
    <span class="mic-dot"></span>
    {{ active ? '停止收音' : '开始语音' }}
  </button>
</template>

