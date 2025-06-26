<template>
    <footer class="app-footer">
      <div class="footer-content container">
        &copy; {{ currentYear }} Dog Breed Detector. All rights reserved.
      </div>
    </footer>
  </template>
  
  <script>
  export default {
    name: 'Footer',
    data() {
      return {
        currentYear: new Date().getFullYear()
      };
    }
  }
  </script>
  
  <style scoped>
  .app-footer {
    background-color: #2c3e50;
    color: white;
    padding: 20px 0;
    text-align: center;
    font-size: 0.9em;
  }
  
  .footer-content {
    padding: 0 20px;
  }
  </style>