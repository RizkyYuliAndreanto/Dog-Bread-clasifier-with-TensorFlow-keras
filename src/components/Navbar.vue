<template>
    <nav class="navbar">
      <div class="navbar-container container">
        <router-link to="/" class="navbar-brand">Dog Detector</router-link>
        <div class="navbar-links">
          <router-link to="/" class="nav-link">Home</router-link>
          <router-link to="/history" class="nav-link">History</router-link>
          <router-link v-if="!isAuthenticated" to="/login" class="nav-link">Login</router-link>
          <router-link v-if="!isAuthenticated" to="/register" class="nav-link">Register</router-link>
          <button v-if="isAuthenticated" @click="logout" class="btn btn-danger nav-button">
            Logout
          </button>
        </div>
      </div>
    </nav>
  </template>
  
  <script>
  export default {
    name: 'Navbar',
    data() {
      return {
        isAuthenticated: false
      };
    },
    created() {
      this.checkAuthStatus();
      window.addEventListener('storage', this.checkAuthStatus);
    },
    beforeUnmount() {
      window.removeEventListener('storage', this.checkAuthStatus);
    },
    methods: {
      checkAuthStatus() {
        // Autentikasi dianggap berhasil jika ada access_token
        this.isAuthenticated = !!localStorage.getItem('access_token');
      },
      logout() {
        localStorage.removeItem('access_token');
        localStorage.removeItem('user_id'); // Hapus juga user_id saat logout
        this.isAuthenticated = false;
        this.$router.push('/login');
        window.dispatchEvent(new Event('storage')); // Memaksa refresh status auth di navbar
      }
    }
  }
  </script>
  
  <style scoped>
  /* Styling tetap sama seperti sebelumnya */
  .navbar {
    background-color: #2c3e50;
    padding: 15px 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  
  .navbar-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .navbar-brand {
    font-size: 1.8em;
    font-weight: bold;
    color: white;
    text-decoration: none;
    letter-spacing: 0.5px;
  }
  
  .navbar-links {
    display: flex;
    align-items: center;
  }
  
  .nav-link {
    color: white;
    text-decoration: none;
    margin-left: 20px;
    font-size: 1.1em;
    transition: color 0.3s ease;
  }
  
  .nav-link:hover {
    color: #a0c4ff;
  }
  
  .nav-button {
    margin-left: 20px;
    padding: 8px 15px;
    font-size: 1em;
  }
  </style>