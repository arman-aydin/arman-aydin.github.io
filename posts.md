---
layout: page
title: Blog
description: >
  Technical articles and deep dives on LLM serving optimization, custom GPU kernels, and deep reinforcement learning.
---

{% for post in site.posts %}
<article class="post-entry" style="margin-bottom: 2rem;">
  <h3 style="margin-bottom: 0.5rem; font-weight: 700;">
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </h3>
  <p style="margin-top: 0; color: var(--text-color-faded); font-size: 0.95rem;">
    {{ post.description }}
  </p>
</article>
{% endfor %}
