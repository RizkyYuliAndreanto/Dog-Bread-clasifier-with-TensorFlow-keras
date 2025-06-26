<template>
    <div class="history-page container">
      <h2 class="page-title">Riwayat Prediksi Anda</h2>
      
      <div v-if="!isAuthenticated" class="alert-message error">
        Anda harus <strong>login</strong> untuk melihat riwayat prediksi Anda.
      </div>
      <div v-else>
        <p v-if="loading" class="loading-message">Memuat riwayat prediksi...</p>
        <p v-else-if="error" class="alert-message error">{{ error }}</p>
        <div v-else-if="history.length === 0" class="no-history-message">
          Belum ada riwayat prediksi yang ditemukan untuk akun ini.
        </div>
        <div v-else class="history-grid">
          <div v-for="item in history" :key="item.id" class="history-item-card">
            <img v-if="item.image_url" :src="backendBaseUrl + item.image_url" :alt="item.predicted_breed" class="history-item-image">
            <div class="history-item-details">
              <p class="history-item-prediction"><strong>Prediksi:</strong> {{ item.predicted_breed }}</p>
              <p class="history-item-confidence"><strong>Keyakinan:</strong> {{ item.confidence }}</p>
              <p class="history-item-timestamp"><strong>Waktu:</strong> {{ formatTimestamp(item.timestamp) }}</p>
              <p class="history-item-user"><strong>Pengguna:</strong> Anda</p> 
            </div>
          </div>
        </div>
      </div>
    </div>
</template>
 
<script>
import apiClient from '@/plugins/axios';

export default {
  name: 'HistoryView',
  data() {
    return {
      history: [],
      loading: true,
      error: '',
      isAuthenticated: false,
      backendBaseUrl: 'http://127.0.0.1:5000'
    };
  },
  created() {
    this.checkAuthStatus();
    window.addEventListener('storage', this.handleStorageChange);

    // Selalu cek ulang token setelah created
    if (this.isAuthenticated) {
      this.fetchHistory();
    } else {
      this.loading = false;
      this.error = 'Autentikasi diperlukan untuk melihat riwayat pribadi Anda.';
    }
  },
  beforeUnmount() {
    window.removeEventListener('storage', this.handleStorageChange);
  },
  methods: {
    checkAuthStatus() {
      // Pastikan token tidak kosong dan bukan string 'undefined'
      const token = localStorage.getItem('access_token');
      this.isAuthenticated = !!token && token !== 'undefined' && token !== 'null';
    },
    handleStorageChange() {
      this.checkAuthStatus();
      if (this.isAuthenticated) {
        this.fetchHistory();
      } else {
        this.history = [];
        this.loading = false;
        this.error = 'Anda harus login untuk melihat riwayat prediksi Anda.';
      }
    },
    async fetchHistory() {
      this.loading = true;
      this.error = '';

      if (!this.isAuthenticated) {
        this.loading = false;
        this.error = 'Anda harus login untuk melihat riwayat prediksi Anda.';
        return;
      }

      try {
        const response = await apiClient.get('/history');
        if (response.status === 200) {
          this.history = response.data;
        } else {
          this.error = response.data.error || 'Gagal memuat riwayat dengan status tidak terduga.';
        }
      } catch (err) {
        this.history = [];
        if (err.response) {
          const status = err.response.status;
          if (status === 401 || status === 403) {
            this.error = err.response.data.message || 'Sesi Anda telah berakhir. Silakan login kembali.';
            localStorage.removeItem('access_token');
            localStorage.removeItem('user_id');
            this.isAuthenticated = false;
          } else if (status === 500) {
            this.error = err.response.data.error || 'Terjadi kesalahan internal server saat mengambil histori.';
          } else {
            this.error = err.response.data.message || err.response.data.error || `Gagal memuat riwayat. Kode status: ${err.response.status}.`;
          }
        } else if (err.request) {
          this.error = 'Tidak ada respons dari server. Periksa koneksi internet Anda atau pastikan server aktif.';
        } else {
          this.error = `Terjadi kesalahan saat memuat riwayat: ${err.message}`;
        }
        console.error('Error fetching history:', err);
      } finally {
        this.loading = false;
      }
    },
    formatTimestamp(timestamp) {
      const date = new Date(timestamp);
      if (isNaN(date.getTime())) {
        return timestamp;
      }
      return date.toLocaleString('id-ID', {
        year: 'numeric', month: 'long', day: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
      });
    }
  },
  watch: {
    isAuthenticated(newValue) {
      if (newValue) {
        this.fetchHistory();
      } else {
        this.history = [];
        this.loading = false;
        this.error = 'Anda harus login untuk melihat riwayat prediksi Anda.';
      }
    }
  }
};
</script>
 
<style scoped>
/* Styling tetap sama seperti sebelumnya */
.history-page {
  padding: 40px 20px;
  min-height: calc(100vh - 170px);
}
 
.page-title {
  font-size: 2.5em;
  margin-bottom: 30px;
  text-align: center;
  color: #333;
}
 
.loading-message,
.no-history-message {
  text-align: center;
  color: #666;
  font-size: 1.1em;
  padding: 20px;
  background-color: #f0f0f0;
  border-radius: 8px;
}
 
.alert-message {
  padding: 15px;
  margin-bottom: 20px;
  border-radius: 5px;
  text-align: center;
}
 
.alert-message.error {
  background-color: #f8d7da;
  color: #721c24;
  border: 1px solid #f5c6cb;
}
 
.history-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 25px;
}
 
.history-item-card {
  background-color: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  transition: transform 0.2s ease;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}
 
.history-item-card:hover {
  transform: translateY(-5px);
}
 
.history-item-image {
  width: 100%;
  max-height: 200px;
  object-fit: cover;
  border-radius: 8px;
  margin-bottom: 15px;
  border: 1px solid #eee;
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
 
.history-item-details {
  width: 100%;
}
 
.history-item-prediction {
  font-size: 1.2em;
  font-weight: 600;
  margin-bottom: 5px;
  color: #007bff;
}
 
.history-item-confidence {
  font-size: 1em;
  color: #4CAF50;
  margin-bottom: 5px;
}
 
.history-item-timestamp {
  font-size: 0.9em;
  color: #777;
}
 
.history-item-user { /* Styling untuk info pengguna */
    font-size: 0.9em;
    color: #555;
    font-style: italic;
    margin-top: 5px;
}
 
@media (max-width: 768px) {
  .history-grid {
    grid-template-columns: 1fr;
  }
}
</style>
