<template>
    <div class="hero-section">
      <div class="hero-content container">
        <h1 class="hero-title">Deteksi Ras Anjing Anda!</h1>
        <p class="hero-subtitle">Upload gambar anjing Anda dan biarkan AI kami mengidentifikasi rasnya.</p>
        <router-link to="/login" class="btn btn-primary hero-button">
          Mulai Sekarang
        </router-link>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    name: 'HeroSection'
  }
  </script>
  
  <style scoped>
  .hero-section {
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%); /* Gradien warna */
    color: white;
    padding: 80px 0;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }
  
  .hero-content {
    max-width: 800px;
    margin: 0 auto;
  }
  
  .hero-title {
    font-size: 3.5em;
    font-weight: bold;
    margin-bottom: 20px;
    line-height: 1.2;
  }
  
  .hero-subtitle {
    font-size: 1.5em;
    margin-bottom: 40px;
    opacity: 0.9;
  }
  
  .hero-button {
    padding: 15px 30px;
    font-size: 1.2em;
    border-radius: 30px; /* Bentuk pil */
    background-color: white;
    color: #2575fc;
    font-weight: bold;
  }
  
  .hero-button:hover {
    background-color: #e0e0e0;
    transform: translateY(-2px);
  }
  </style>