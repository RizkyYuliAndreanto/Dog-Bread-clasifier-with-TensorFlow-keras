import axios from "axios";

const apiClient = axios.create({
  baseURL: "http://127.0.0.1:5000/api",
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 10000,
});

apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("access_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
      console.log("Authorization header set:", config.headers.Authorization); // Tambahkan log ini
    } else {
      console.log("No access_token found in localStorage");
    }
    return config;
  },
  (error) => Promise.reject(error)
);

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Tangani error otentikasi global jika token hilang/tidak valid/kadaluarsa
    if (
      error.response &&
      (error.response.status === 401 || error.response.status === 403)
    ) {
      console.warn(
        "Token JWT mungkin tidak valid atau kadaluarsa. Mengarahkan ke Login."
      );
      localStorage.removeItem("access_token");
      localStorage.removeItem("user_id"); // Hapus juga user_id
      // Memancarkan event storage untuk memperbarui Navbar
      window.dispatchEvent(new Event("storage"));
      // Penting: Di sini kita tidak bisa langsung `this.$router.push('/login')`
      // karena interceptor tidak memiliki akses ke instance Vue router.
      // Solusi terbaik adalah menangani redirect ini di komponen yang memanggil API,
      // atau menggunakan sistem event bus global/Vuex untuk memicu redirect.
      // Untuk demo ini, kita akan biarkan komponen masing-masing menangani redirect.
    }
    return Promise.reject(error);
  }
);

export default apiClient;
