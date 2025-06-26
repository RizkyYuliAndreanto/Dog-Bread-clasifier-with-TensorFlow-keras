<template>
    <div class="register-page">
      <div class="register-card">
        <h2 class="register-title">Daftar Akun Baru</h2>
        <form @submit.prevent="register">
          <div class="form-group">
            <label for="name">Nama Lengkap:</label>
            <input
              type="text"
              id="name"
              v-model="name"
              class="form-input"
              required
              aria-describedby="name-help"
              autocomplete="name"
            />
            <small id="name-help" class="input-help-text">Masukkan nama lengkap Anda.</small>
          </div>
  
          <div class="form-group">
            <label for="username">Username:</label>
            <input
              type="text"
              id="username"
              v-model="username"
              class="form-input"
              required
              aria-describedby="username-help"
              autocomplete="username"
            />
            <small id="username-help" class="input-help-text">Gunakan kombinasi huruf dan angka.</small>
          </div>
  
          <div class="form-group">
            <label for="password">Password:</label>
            <input
              type="password"
              id="password"
              v-model="password"
              class="form-input"
              required
              aria-describedby="password-help"
              autocomplete="new-password"
            />
            <small id="password-help" class="input-help-text">Minimal 6 karakter.</small>
          </div>
          
          <button
            type="submit"
            class="btn btn-primary btn-block"
            :disabled="isRegistering"
          >
            {{ isRegistering ? 'Mendaftar...' : 'Daftar' }}
          </button>
  
          <p v-if="errorMessage" class="error-message" role="alert">{{ errorMessage }}</p>
  
          <p class="auth-link-text">
            Sudah punya akun?
            <router-link to="/login" class="auth-link">Login di sini</router-link>
          </p>
        </form>
      </div>
    </div>
  </template>
  
  <script>
  import apiClient from '@/plugins/axios';
  
  export default {
    name: 'RegisterView',
    data() {
      return {
        name: '',
        username: '',
        password: '',
        errorMessage: '',
        isRegistering: false
      };
    },
    methods: {
      async register() {
        this.errorMessage = '';
        this.isRegistering = true;
  
        try {
          const response = await apiClient.post('/auth/register', {
            username: this.username,
            password: this.password,
            name: this.name
          });
          
          // Dokumentasi API menyatakan response sukses 201 dengan { "message": "..." }
          if (response.status === 201) {
            alert(response.data.message || 'Registrasi berhasil! Silakan login dengan akun Anda.');
            this.$router.push('/login');
          } else {
            // Fallback jika backend mengembalikan status 2xx tapi bukan 201
            this.errorMessage = response.data.message || 'Registrasi gagal dengan status yang tidak terduga.';
          }
          
        } catch (error) {
          if (error.response) {
            // Backend merespons dengan status error
            if (error.response.status === 409) {
              this.errorMessage = error.response.data.error || 'Username ini sudah terdaftar. Mohon gunakan username lain.';
            } else if (error.response.status === 400) {
              this.errorMessage = error.response.data.message || 'Data tidak lengkap. Username, password, dan nama lengkap wajib diisi.';
            } else {
              // Error lain dari backend
              this.errorMessage = error.response.data.message || error.response.data.error || `Registrasi gagal. Kode status: ${error.response.status}.`;
            }
          } else if (error.request) {
            // Request dibuat tetapi tidak ada respons (misal server mati)
            this.errorMessage = 'Tidak ada respons dari server. Periksa koneksi internet Anda atau pastikan server aktif.';
          } else {
            // Kesalahan lain yang tidak terkait dengan respons HTTP
            this.errorMessage = `Terjadi kesalahan saat registrasi: ${error.message}`;
          }
          console.error('Error during registration:', error);
        } finally {
          this.isRegistering = false;
        }
      }
    }
  }
  </script>
  
  <style scoped>
  /* Styling sama seperti sebelumnya */
  .register-page {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 40px 20px;
    min-height: calc(100vh - 170px);
    background-color: #eef2f6;
  }
  
  .register-card {
    background-color: white;
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    width: 100%;
    max-width: 480px;
    box-sizing: border-box;
  }
  
  .register-title {
    font-size: 2.5em;
    font-weight: 700;
    margin-bottom: 30px;
    text-align: center;
    color: #333;
  }
  
  .form-group {
    margin-bottom: 20px;
  }
  
  .form-group label {
    display: block;
    margin-bottom: 8px;
    font-weight: 600;
    color: #555;
  }
  
  .form-input {
    width: 100%;
    padding: 12px;
    border: 1px solid #ced4da;
    border-radius: 6px;
    box-sizing: border-box;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }
  
  .form-input:focus {
    outline: none;
    border-color: #007bff;
    box-shadow: 0 0 0 0.25rem rgba(0, 123, 255, 0.25);
  }
  
  .input-help-text {
    font-size: 0.85em;
    color: #6c757d;
    margin-top: 5px;
    display: block;
  }
  
  .btn-block {
    width: 100%;
    margin-top: 25px;
    padding: 12px 20px;
    font-size: 1.1em;
    border-radius: 6px;
  }
  
  .error-message {
    color: #dc3545;
    font-size: 0.9em;
    margin-top: 20px;
    text-align: center;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    padding: 10px;
    border-radius: 5px;
  }
  
  .auth-link-text {
    text-align: center;
    margin-top: 25px;
    font-size: 0.95em;
    color: #666;
  }
  
  .auth-link {
    color: #007bff;
    text-decoration: none;
    font-weight: 600;
  }
  
  .auth-link:hover {
    text-decoration: underline;
  }
  </style>