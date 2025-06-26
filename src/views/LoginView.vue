<template>
  <div class="auth-page">
    <div class="auth-card">
      <h2 class="auth-title">Login ke Akun Anda</h2>
      <form @submit.prevent="login">
        <div class="form-group">
          <label for="username">Username:</label>
          <input
            type="text"
            id="username"
            v-model="username"
            class="form-input"
            required
            autocomplete="username"
          />
        </div>
        <div class="form-group">
          <label for="password">Password:</label>
          <input
            type="password"
            id="password"
            v-model="password"
            class="form-input"
            required
            autocomplete="current-password"
          />
        </div>
        <button
          type="submit"
          class="btn btn-success btn-block"
          :disabled="isLoggingIn"
        >
          {{ isLoggingIn ? 'Logging In...' : 'Login' }}
        </button>
        <p v-if="errorMessage" class="error-message">{{ errorMessage }}</p>
        <p class="auth-link-text">
          Belum punya akun?
          <router-link to="/register" class="auth-link">Daftar di sini</router-link>
        </p>
      </form>
    </div>
  </div>
</template>

<script>
import apiClient from '@/plugins/axios';

export default {
  name: 'LoginView',
  data() {
    return {
      username: '',
      password: '',
      errorMessage: '',
      isLoggingIn: false
    };
  },
  methods: {
    async login() {
      this.errorMessage = '';
      this.isLoggingIn = true;

      try {
        const response = await apiClient.post('/auth/login', {
          username: this.username,
          password: this.password
        });

        if (response.status === 200 && response.data.access_token) {
          localStorage.setItem('access_token', response.data.access_token);
          localStorage.setItem('user_id', response.data.user_id);

          // Debug log untuk memastikan token tersimpan
          console.log('Token disimpan:', localStorage.getItem('access_token'));

          // Trigger event storage agar komponen lain tahu status login berubah
          window.dispatchEvent(new Event('storage'));

          alert(response.data.message || 'Login berhasil!');
          this.$router.push('/');
        } else {
          this.errorMessage = response.data.message || 'Login gagal dengan status yang tidak terduga.';
        }
      } catch (error) {
        if (error.response) {
          if (error.response.status === 401) {
            this.errorMessage = error.response.data.error || 'Username atau password salah.';
          } else if (error.response.status === 400) {
            this.errorMessage = error.response.data.message || 'Username dan password wajib diisi.';
          } else {
            this.errorMessage = error.response.data.message || error.response.data.error || `Login gagal. Kode status: ${error.response.status}.`;
          }
        } else if (error.request) {
          this.errorMessage = 'Tidak ada respons dari server. Periksa koneksi internet Anda atau pastikan server aktif.';
        } else {
          this.errorMessage = `Terjadi kesalahan saat login: ${error.message}`;
        }
        console.error('Error during login:', error);
      } finally {
        this.isLoggingIn = false;
      }
    }
  }
}
</script>

<style scoped>
/* Styling sama seperti sebelumnya */
.auth-page {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 40px 20px;
  min-height: calc(100vh - 170px);
  background-color: #eef2f6;
}

.auth-card {
  background-color: white;
  padding: 30px;
  border-radius: 10px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: 480px;
  box-sizing: border-box;
}

.auth-title {
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